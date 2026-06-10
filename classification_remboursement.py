"""
Classifieur multi-classes (5 catégories) pour les UCD issues de fichiers RDF (T2A et ATU). 
Il distingue les sous-types ATU (AAC, AAD, AAP, CPC) de la catégorie T2A (LES-MCO).

Fonctionnement :
--------
1. Lecture des graphes T2A et ATU via rdflib, extraction des libellés, des codes indication 
    nationale et des labels (marqueurs [AAX] ou source T2A)
2. Suppression des marqueurs [AAC/AAD/AAP/CPC/LES-MCO] en fin de libellé avant le split 
    pour éviter tout data leakage
3. Création des features : 3 features textuelles (longueur, nombre de mots, majuscules) +
   4 features sur le code indication nationale (chiffres, initiale C/N/I).
4. Prétraitement : TF-IDF trigrammes (word-level) sur libelleLong via ColumnTransformer,
   fitté uniquement sur le train.
5. Création des baselines : DummyClassifier (stratified + most_frequent) 
6. Optimisation grace à Optuna qui va chercher les meilleures hyperparamètres XGBoost
     avec validation croisée StratifiedKFold 5 folds et pondération des classes.
7. Evaluation et sauvegarde

"""
import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from scipy.stats import chi2_contingency
import xgboost as xgb
from optuna import create_study, samplers
from optuna.trial import Trial
from rdflib import Graph, URIRef
import matplotlib.pyplot as plt
import seaborn as sns


# Constantes
LABEL_T2A = "LES-MCO"
ATU_VALID_SUBTYPES = {"AAC", "AAD", "AAP", "CPC"}
ALL_VALID_LABELS = ATU_VALID_SUBTYPES | {LABEL_T2A}
PROP_LIBELLE_LONG = URIRef("http://exempleURI/libellelong")
PROP_ROLE_UCD = URIRef("http://exempleURI/roleUCD")
PROP_ROLE_INDICATION_JP = URIRef("http://exempleURI/roleIndicationJP")
PROP_ROLE_INDICATION_NATIONALE = URIRef("http://exempleURI/roleIndicationNationale")

# Features
TEXT_COLS = ["libelleLong"]
NUM_COLS  = [
    # Libellé (3 features)
    "libelleLong_length",
    "libelle_num_words",
    "libelle_num_chars_upper",
    # Code indication nationale (4 features)
    "indication_nat_num_digits",
    "indication_nat_starts_C",
    "indication_nat_starts_N",
    "indication_nat_starts_I",
]
TARGET = "categorie"


# Utilitaires
def clean_libelle(text: str) -> str:
    """
    Supprime les marqueurs [AAC], [AAD], [AAP], [CPC], [LES-MCO] en fin de libellé
    Appliqué avant le split pour eviter du data leakage
    """
    return re.sub(r'\s*\[[^\]]+\]\s*$', '', str(text)).strip()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les 8 features numériques : 3 du libellé + 4 du code indication nationale
    A appeler après clean_libelle.
    """
    df = df.copy()

    # Libellé (3 features)
    df["libelleLong_length"]= df["libelleLong"].apply(len)
    df["libelle_num_words"] = df["libelleLong"].apply(lambda x: len(x.split()))
    df["libelle_num_chars_upper"] = df["libelleLong"].apply(lambda x: sum(c.isupper() for c in x))

    # Code indication nationale (4 features)
    ind_nat = df["indication_nationale"].fillna("").astype(str)
    ind_nat_first = ind_nat.str.upper().str[:1]
    df["indication_nat_num_digits"] = ind_nat.apply(lambda x: sum(c.isdigit() for c in x))
    df["indication_nat_starts_C"] = (ind_nat_first == "C").astype(int)
    df["indication_nat_starts_N"] = (ind_nat_first == "N").astype(int)
    df["indication_nat_starts_I"] = (ind_nat_first == "I").astype(int)
    return df

def to_dense(X) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else X

# Chargement RDF
def _extract_atu_subtype(libelle: str, uri: str = "") -> str:
    """
    Extrait le sous-type ATU depuis le libellé original (avant nettoyage)
    Utilisé uniquement pour labelliser la ligne
    La grosse différence entre AAC, AAD, AAP et CPC est que la catégorie CPC possède un attribut en 
    plus 'aPourModalitePriseEnChargeMCO'donc forcement si l'UCD possède cet attribut alors il est forcement CPC
    """
    match = re.search(r'\[([^\]]+)\]\s*$', str(libelle).strip())
    if match:
        candidate = match.group(1).strip().upper()
        if candidate in ATU_VALID_SUBTYPES:
            return candidate
    if "aPourModalitePriseEnChargeMCO" in str(uri):
        return "CPC"
    return "INCONNU"


def load_rdf_data(rdf_path: str, source: str) -> pd.DataFrame:
    """
    Charge un fichier RDF et retourne un DataFrame avec les features brutes
    Pour les ATU, le label est extrait depuis le marqueur [AAC/AAD/AAP/CPC] qui est présent dans 
    le libellé (les marqueurs sont supprimés ensuite par clean_libelle)
    """
    g = Graph()
    g.parse(rdf_path, format="xml")

    logging.info(f"{len(g)} triplets chargés depuis {rdf_path}")

    data = []
    for subj in g.subjects():
        libelle = g.value(subj, PROP_LIBELLE_LONG, None)
        roleUCD= g.value(subj, PROP_ROLE_UCD, None)
        roleJP= g.value(subj, PROP_ROLE_INDICATION_JP, None)
        roleNAT = g.value(subj, PROP_ROLE_INDICATION_NATIONALE, None)

        if libelle is None:
            continue

        if source == "T2A":
            label = LABEL_T2A
        else:
            label = _extract_atu_subtype(libelle, uri=str(subj))
            if label == "INCONNU":
                continue

        data.append({
            "uri": str(subj),
            "libelleLong":  str(libelle),
            "indication_nationale": str(roleNAT) if roleNAT else "",
            "categorie": label,
        })

    logging.info(f"{len(data)} entrées retenues (source={source})")
    return pd.DataFrame(data)


# Préprocesseur
def build_preprocessor(max_features_libelle: int = 5000) -> ColumnTransformer:
    """
    Construit le ColumnTransformer (TF-IDF trigrammes sur libelleLong )
    """
    tfidf_libelle = TfidfVectorizer(max_features=max_features_libelle,ngram_range=(1, 3),sublinear_tf=True,min_df=2,analyzer="word",)
    transformers = [("tfidf_libelle", tfidf_libelle, "libelleLong")]
    return ColumnTransformer(transformers, remainder="drop")


def apply_preprocessor(preprocessor: ColumnTransformer,df: pd.DataFrame,fit: bool = False) -> np.ndarray:
    """
    Applique le préprocesseur TF-IDF et concatène les features numériques
    """
    if fit:
        X_sparse = preprocessor.fit_transform(df[TEXT_COLS])
    else:
        X_sparse = preprocessor.transform(df[TEXT_COLS])

    X_num = df[NUM_COLS].values.astype(float)
    return np.hstack([to_dense(X_sparse), X_num])


# Baseline
def evaluate_baseline(X_train: np.ndarray, y_train: pd.Series, X_test: np.ndarray, y_test: pd.Series, output_dir: Path) -> None:
    results = []
    for strategy in ("stratified", "most_frequent"):
        dummy = DummyClassifier(strategy=strategy, random_state=42)
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        f1 = f1_score(y_test, y_pred_dummy, average="macro", zero_division=0)
        present_labels = sorted(set(y_test) | set(y_pred_dummy))
        report = classification_report(
            y_test, y_pred_dummy,
            labels=present_labels,
            target_names=present_labels,
            zero_division=0,
        )
        results.append(f"Baseline : {strategy} (macro F1 = {f1:.4f}) \n{report}\n")
        logging.info(f"Baseline {strategy} : macro F1 = {f1:.4f}")

    path = output_dir / "baseline_report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    logging.info(f"Baseline sauvegardée dasn {path}")

# Entrainement
def train_model(X_train: np.ndarray, y_train: pd.Series,class_weights: Dict, n_trials: int = 40) -> Tuple:
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    sample_weights  = np.array([class_weights[label] for label in y_train])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial: Trial) -> float:
        params = {
            "n_estimators":trial.suggest_int("n_estimators",100, 500),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree",0.6, 1.0),
            "gamma": trial.suggest_float("gamma",0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":trial.suggest_float("reg_lambda",0.0, 1.0),
            "min_child_weight": trial.suggest_int(  "min_child_weight",1, 10),
            "random_state": 42,
            "eval_metric":"mlogloss",
            "verbosity": 0,
        }
        model = xgb.XGBClassifier(**params)
        fold_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train_encoded):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
            sw_tr = sample_weights[train_idx]
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
            y_val_pred  = model.predict(X_val)
            fold_scores.append(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
        return float(np.mean(fold_scores))

    study = create_study(direction="maximize", sampler=samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"random_state": 42, "eval_metric": "mlogloss", "verbosity": 0})
    logging.info(f"Meilleurs hyperparamètres (CV mean F1={study.best_value:.4f}) : {best_params}")

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train_encoded, sample_weight=sample_weights)

    # Recalcul des scores par fold avec les meilleurs params (pour boxplot)
    fold_scores_best = []
    for train_idx, val_idx in cv.split(X_train, y_train_encoded):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[val_idx]
        sw_tr = sample_weights[train_idx]
        m = xgb.XGBClassifier(**best_params)
        m.fit(X_tr, y_tr, sample_weight=sw_tr)
        fold_scores_best.append(f1_score(m.predict(X_val), y_val, average="macro", zero_division=0))

    return model, le, study, fold_scores_best


# Visualisations
def plot_confusion_matrix(y_true, y_pred, labels: List[str], output_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",xticklabels=labels, yticklabels=labels)
    plt.title("Matrice de confusion — classification IndicationJP (5 classes)")
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Matrice de confusion {path}")


def plot_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    distribution = df[TARGET].value_counts()
    total = distribution.sum()
    labels_list  = distribution.index.tolist()
    counts= distribution.values.tolist()
    pcts = [c / total * 100 for c in counts]

    colors = {
        "LES-MCO": "#185FA5", "AAC": "#4A90E2",
        "AAP":     "#7BB3F0", "CPC": "#AACCFF", "AAD": "#D4E6FF",
    }
    bar_colors = [colors.get(l, "#888888") for l in labels_list]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels_list[::-1], counts[::-1], color=bar_colors[::-1], height=0.55)
    for bar, count, pct in zip(bars, counts[::-1], pcts[::-1]):
        ax.text(
            bar.get_width() + total * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({pct:.1f}%)",
            va="center", ha="left", fontsize=11, color="#444444",
        )
    ax.set_xlabel("Effectif", fontsize=11, color="#5F5E5A")
    ax.set_title("Distribution des classes — IndicationJP", fontsize=13,fontweight="normal", pad=14)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=11)
    ax.tick_params(axis="x", colors="#888888")
    ax.set_xlim(0, total * 1.22)
    ax.grid(axis="x", linestyle="--", alpha=0.4, color="#D3D3D3")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    path = output_dir / "distribution_classes.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Distribution {path}")


def plot_feature_importance(model: xgb.XGBClassifier, preprocessor: ColumnTransformer, output_dir: Path) -> None:
    importances   = model.feature_importances_
    feature_names: List[str] = []
    for name, transformer, _ in preprocessor.transformers_:
        if hasattr(transformer, "get_feature_names_out"):
            feature_names.extend([f"{name}__{fn}" for fn in transformer.get_feature_names_out()])
        else:
            feature_names.extend([f"{name}__{i}" for i in range(transformer.n_features_in_)])
    feature_names.extend(NUM_COLS)

    n   = min(30, len(importances))
    idx = np.argsort(importances)[-n:]
    names_top = [
        feature_names[i] if i < len(feature_names) else f"feat_{i}"
        for i in idx
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(n), importances[idx], color="#4A90E2")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names_top, fontsize=8)
    ax.set_title("Top-30 features importantes (XGBoost)", fontsize=12)
    ax.set_xlabel("Importance (gain)")
    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Feature importance {path}")


def plot_optuna_history(study, output_dir: Path) -> None:
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = [max(values[:i + 1]) for i in range(len(values))]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(values, alpha=0.4, color="#4A90E2", label="F1 trial")
    ax.plot(best_so_far, color="#185FA5", label="Meilleur F1")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Macro F1 (CV)")
    ax.set_title("Convergence Optuna")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = output_dir / "optuna_convergence.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Courbe Optuna {path}")


def plot_learning_curve(model: xgb.XGBClassifier,le: LabelEncoder,X_train: np.ndarray,y_train: pd.Series,output_dir: Path, cv: int = 5,) -> None:
    """
    Courbe d'apprentissage : F1 macro train vs validation en fonction de la taille du jeu d'entrainement
    """
    # XGBoost attend des entiers — on réutilise le LabelEncoder fitté dans train_model
    y_train_enc = le.transform(y_train)
    n_classes = len(le.classes_)

    # On clone le modèle en forçant num_class pour éviter l'erreur sur les folds
    # qui ne contiennent pas toutes les classes (ex : AAD avec seulement 3 exemples)
    lc_params = model.get_params()
    lc_params["num_class"] = n_classes
    lc_model  = xgb.XGBClassifier(**lc_params)

    train_sizes, train_scores, val_scores = learning_curve(
        lc_model, X_train, y_train_enc,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring="f1_macro",
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=1,
        error_score=0.0,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_mean, "o-", color="#185FA5", label="Train")
    ax.fill_between(train_sizes,train_mean - train_std, train_mean + train_std,alpha=0.15, color="#185FA5")
    ax.plot(train_sizes, val_mean, "o-", color="#E2703A", label="Validation (CV)")
    ax.fill_between(train_sizes,val_mean - val_std, val_mean + val_std,alpha=0.15, color="#E2703A")
    ax.set_xlabel("Taille du jeu d'entraînement", fontsize=11)
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_title("Courbe d'apprentissage", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = output_dir / "learning_curve.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Courbe d'apprentissage {path}")


def plot_train_test_gap(model: xgb.XGBClassifier,le: LabelEncoder,X_train: np.ndarray,y_train: pd.Series,X_test: np.ndarray,y_test: pd.Series,output_dir: Path,) -> None:
    """
    Barplot F1 par classe : train vs test
    """
    labels = sorted(ALL_VALID_LABELS)

    y_train_pred = le.inverse_transform(model.predict(X_train))
    y_test_pred = le.inverse_transform(model.predict(X_test))

    from sklearn.metrics import classification_report as cr
    def f1_per_class(y_true, y_pred):
        rep = cr(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
        return [rep.get(l, {}).get("f1-score", 0.0) for l in labels]

    train_f1s = f1_per_class(y_train, y_train_pred)
    test_f1s = f1_per_class(y_test,  y_test_pred)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_tr = ax.bar(x - width / 2, train_f1s, width, label="Train",color="#185FA5", alpha=0.85)
    bars_te = ax.bar(x + width / 2, test_f1s,  width, label="Test",color="#E2703A", alpha=0.85)

    for bar in bars_tr:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color="#185FA5")
    for bar in bars_te:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color="#E2703A")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("F1 par classe : Train vs Test (détection de surapprentissage)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = output_dir / "train_test_gap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Train/Test gap {path}")


def plot_cv_boxplot(fold_scores: List[float], output_dir: Path) -> None:
    """
    Boxplot des F1 macro sur les 5 folds de validation croisée.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot(fold_scores, patch_artist=True, widths=0.4,medianprops=dict(color="#E2703A", linewidth=2))
    bp["boxes"][0].set_facecolor("#AED0F5")

    for i, score in enumerate(fold_scores):
        ax.scatter(1, score, color="#185FA5", zorder=5, s=50)
        ax.text(1.07, score, f"Fold {i+1}: {score:.3f}",va="center", fontsize=9, color="#444444")

    mean_val = np.mean(fold_scores)
    std_val  = np.std(fold_scores)
    ax.axhline(mean_val, color="#185FA5", linestyle="--", linewidth=1.2,label=f"Moyenne = {mean_val:.3f} ± {std_val:.3f}")

    ax.set_xticks([1])
    ax.set_xticklabels(["XGBoost (best params)"])
    ax.set_ylabel("Macro F1", fontsize=11)
    ax.set_ylim(max(0, min(fold_scores) - 0.05), min(1.05, max(fold_scores) + 0.1))
    ax.set_title("Variance inter-folds (CV 5-folds)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = output_dir / "cv_boxplot.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Boxplot CV {path}")


def run_mcnemar_test(y_test: pd.Series,y_pred_model: np.ndarray,y_pred_baseline: np.ndarray,output_dir: Path,) -> None:
    """
    Test de McNemar : compare XGBoost à la baseline 'most_frequent'.
    Statistique chi2 = (b - c)² / (b + c)
    """
    correct_model = (y_pred_model == y_test.values)
    correct_baseline = (y_pred_baseline == y_test.values)

    b = int(( correct_model & ~correct_baseline).sum())  # XGB OK, baseline KO
    c = int((~correct_model &  correct_baseline).sum())  # XGB KO, baseline OK

    if b + c == 0:
        logging.warning("McNemar : b+c=0, les deux modèles font exactement les mêmes erreurs.")
        return

    # Table 2x2 pour chi2_contingency (approx. McNemar)
    table = np.array([[0, b], [c, 0]])
    chi2, p_value, _, _ = chi2_contingency(table, correction=True)

    significance = "Significatif (p < 0.05)" if p_value < 0.05 else " Non sign (p ≥ 0.05)"
    result_lines = [
        "Test de McNemar : XGBoost vs Baseline (most_frequent) ",
        f"  b (XGB correct, baseline faux) = {b}",
        f"  c (XGB faux, baseline correct) = {c}",
        f"  chi2 = {chi2:.4f}",
        f"  p-value = {p_value:.6f}",
        f"  {significance}",
    ]
    result_text = "\n".join(result_lines)
    logging.info(result_text)

    path = output_dir / "mcnemar_test.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(result_text + "\n")
    logging.info(f"Test McNemar {path}")

    # Visualisation de la table de contingence
    fig, ax = plt.subplots(figsize=(5, 4))
    table_display = np.array([[f"—", str(b)], [str(c), "—"]])
    ax.axis("off")
    tbl = ax.table(
        cellText=table_display,
        rowLabels=["XGB correct", "XGB faux"],
        colLabels=["Baseline correct", "Baseline faux"],
        cellLoc="center", loc="center",
    )
    tbl.scale(1.4, 2.0)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    ax.set_title(
        f"McNemar  chi2={chi2:.3f}  p={p_value:.4f}\n{significance}",
        fontsize=11, pad=20,
    )
    plt.tight_layout()
    path_fig = output_dir / "mcnemar_table.png"
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Table McNemar {path_fig}")


def plot_permutation_test(model: xgb.XGBClassifier,le: LabelEncoder,X_test: np.ndarray,y_test: pd.Series,output_dir: Path,n_permutations: int = 500,) -> None:
    """
    Test de significativité par permutation 
    """
    y_test_enc = le.transform(y_test)
    y_pred_enc = model.predict(X_test)
    observed_f1 = f1_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)

    rng= np.random.default_rng(42)
    null_scores = np.array([
        f1_score(rng.permutation(y_test_enc), y_pred_enc, average="macro", zero_division=0)
        for _ in range(n_permutations)
    ])

    p_value = (null_scores >= observed_f1).mean()
    significance = "Significatif (p < 0.05)" if p_value < 0.05 else "Non sign (p ≥ 0.05)"

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(null_scores, bins=40, color="#AACCFF", edgecolor="white",label=f"Distribution H0 ({n_permutations} permutations)")
    ax.axvline(observed_f1, color="#185FA5", linewidth=2.5,label=f"Score observé = {observed_f1:.4f}  (p = {p_value:.4f})")
    ax.set_xlabel("Macro F1", fontsize=11)
    ax.set_ylabel("Fréquence", fontsize=11)
    ax.set_title(f"Test de significativité par permutation  (n={n_permutations})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.4)
    ax.text(0.98, 0.95, significance, transform=ax.transAxes,ha="right", va="top", fontsize=10,color="#185FA5" if p_value < 0.05 else "#CC0000", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"))
    plt.tight_layout()
    path = output_dir / "permutation_test.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logging.info(f"Test de permutation {path}  (p-value = {p_value:.4f})")

    # Sauvegarde du résultat textuel
    result = (
        f"Test de permutation (n={n_permutations}) \n"
        f"  Score observé (macro F1) = {observed_f1:.4f}\n"
        f"  p-value empirique = {p_value:.4f}\n"
        f" {significance}\n"
    )
    with open(output_dir / "permutation_test.txt", "w", encoding="utf-8") as f:
        f.write(result)


# Main
def main():
    parser = argparse.ArgumentParser(description="Classifieur 5 classes : LES-MCO | AAC | AAD | AAP | CPC")
    parser.add_argument("rdf_t2a",  help="Fichier RDF T2A ")
    parser.add_argument("rdf_atu",  help="Fichier RDF ATU ")
    parser.add_argument("--output", default="output", help="Répertoire de sortie")
    parser.add_argument("--trials", type=int, default=40,help="Nombre de trials Optuna (défaut : 40)")
    parser.add_argument("--max-features", type=int, default=5000,help="Max features TF-IDF libelleLong (défaut : 5000)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    #Chargement 
    logging.info("Chargement des données T2A …")
    df_t2a = load_rdf_data(args.rdf_t2a, source="T2A")
    logging.info(f"  {len(df_t2a)} entrées T2A chargées")

    logging.info("Chargement des données ATU …")
    df_atu = load_rdf_data(args.rdf_atu, source="ATU")
    logging.info(f"  {len(df_atu)} entrées ATU chargées")

    df = pd.concat([df_t2a, df_atu], ignore_index=True)
    logging.info(f"Dataset total : {len(df)} entrées")

    if len(df) == 0:
        logging.error("Aucune donnée chargée.")
        return

    # Nettoyage des marqueurs avant le split 
    logging.info("Suppression des marqueurs [AAX] dans libelleLong…")
    df["libelleLong"] = df["libelleLong"].apply(clean_libelle)

    #Feature enginering
    df = add_engineered_features(df)

    # Distribution
    plot_distribution(df, output_dir)
    logging.info(f"Distribution des classes :\n{df[TARGET].value_counts()}")

    # Split stratifié avant le fit du préproesseur 
    y = df[TARGET]
    X_df_train, X_df_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)
    X_df_train = X_df_train.reset_index(drop=True)
    X_df_test= X_df_test.reset_index(drop=True)
    y_train= y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    logging.info(f"Train : {len(X_df_train)} | Test : {len(X_df_test)}")
    logging.info(f"Distribution train :\n{y_train.value_counts()}")
    logging.info(f"Distribution test  :\n{y_test.value_counts()}")

    #Préprocesseur fitté que sur le train 
    preprocessor = build_preprocessor(max_features_libelle=args.max_features)
    X_train = apply_preprocessor(preprocessor, X_df_train, fit=True)
    X_test = apply_preprocessor(preprocessor, X_df_test, fit=False)

    logging.info(f"Dimension X_train : {X_train.shape}")
    logging.info(f"Dimension X_test: {X_test.shape}")

    #Poids de classe 
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    logging.info(f"Poids de classe : {class_weights}")

    #Baselines 
    logging.info("Calcul des baselines…")
    evaluate_baseline(X_train, y_train, X_test, y_test, output_dir)

    # Entrainement 
    logging.info("Optimisation Optuna (StratifiedKFold 5 folds) + XGBoost…")
    model, le, study, fold_scores_best = train_model(X_train, y_train, class_weights, n_trials=args.trials)

    # Evaluation
    y_pred_encoded = model.predict(X_test)
    y_pred = le.inverse_transform(y_pred_encoded)

    report = classification_report(y_test, y_pred,target_names=sorted(ALL_VALID_LABELS),zero_division=0,)
    print(report)

    support = y_test.value_counts()
    for cls in ALL_VALID_LABELS:
        n = support.get(cls, 0)
        if n < 10:
            logging.warning(
                f"  Classe '{cls}' : seulement {n} exemples en test — "
                f"métriques non significatives statistiquement."
            )

    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logging.info(f"Rapport {report_path}")

    # Erreurs 
    mask_errors = y_test.values != y_pred
    errors_df = X_df_test.copy()
    errors_df["vraie_categorie"] = y_test.values
    errors_df["categorie_predite"] = y_pred
    errors_df = errors_df[mask_errors]

    errors_path = output_dir / "errors.csv"
    try:
        errors_df.to_csv(errors_path, index=False, encoding="utf-8")
        logging.info(f"Erreurs{errors_path}")
    except PermissionError:
        logging.warning(f"Impossible d'écrire {errors_path}")

    # Visualisations 
    plot_confusion_matrix(y_test, y_pred, labels=sorted(ALL_VALID_LABELS), output_dir=output_dir)
    plot_feature_importance(model, preprocessor, output_dir)
    plot_optuna_history(study, output_dir)
    plot_learning_curve(model, le, X_train, y_train, output_dir)
    plot_train_test_gap(model, le, X_train, y_train, X_test, y_test, output_dir)
    plot_cv_boxplot(fold_scores_best, output_dir)
    plot_permutation_test(model, le, X_test, y_test, output_dir)

    # Sauvegarde du modèle 
    model_path = output_dir / "classifier_5classes.joblib"
    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessor,
            "label_encoder": le,
            "text_cols": TEXT_COLS,
            "num_cols": NUM_COLS,
        },
        model_path,
    )
    logging.info(f"Modèle sauvegardé dans {model_path}")
    logging.info(f"\n Tous les fichiers sont dans : {output_dir}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()