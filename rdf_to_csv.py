import rdflib
import pandas
import time
import csv
import argparse
import sys


def parse_rdf(nom_fichier):
    print("Parsing RDF File...")
    graph = rdflib.Graph()
    graph.parse(nom_fichier)
    print("Parsing fait")
    return graph


def fetch_uris(graph):
    print("Fetching all associations URIs")
    uris = []
    selectQuery = graph.query("""
    SELECT DISTINCT ?uri
    WHERE {
           ?uri a <http://exemple/URI1>
    }                          
    """)
    for row in selectQuery:
        uris.append(row["uri"])
    return uris


def process_uri(graph, uri):
    """
    Extrait toutes les données associées à une URI et retourne un dict de ligne CSV, ou None si à ignorer.
    """
    subject = uri

    role_indication_jp = graph.value(subject, rdflib.URIRef("http://exemple/URI2"))
    role_UCD = graph.value(subject, rdflib.URIRef("http://exemple/URI3"))
    role_LES = graph.value(subject, rdflib.URIRef("http://exemple/URI3"))
    date_debut_php1ref = graph.value(subject, rdflib.URIRef("http://exemple/URI3"))
    date_fin_php1ref = graph.value(subject, rdflib.URIRef("http://exemple/URI3"))
    statut_T2A_ATU = graph.value(subject, rdflib.URIRef("http://exemple/URI3"))

    classification_jpucdmo_statut_T2A_ATU = str(statut_T2A_ATU) if statut_T2A_ATU else ""

    role_indication_jp_code = role_indication_jp.split("/")[-1] if role_indication_jp else ""

    role_indication_jp_libelle = ""
    if role_indication_jp:
        libelle_subject = graph.value(role_indication_jp, rdflib.URIRef("http://exemple/URI4"))
        if libelle_subject:
            role_indication_jp_libelle = str(libelle_subject)

    role_ucd_code = role_UCD.split("/")[-1] if role_UCD else ""

    role_ucd_libelle = ""
    date_date_fin_statut_t2a = ""
    if role_UCD:
        libelle_ucd_subject = graph.value(role_UCD, rdflib.URIRef("http://exemple/URI5"))
        date_fin_statut_t2a = graph.value(role_UCD, rdflib.URIRef("http://exemple/URI6"))
        if libelle_ucd_subject:
            role_ucd_libelle = str(libelle_ucd_subject)
            if date_fin_statut_t2a:
                date_date_fin_statut_t2a = str(date_fin_statut_t2a)

    if role_LES:
        role_les_code = role_LES.split("/")[-1]
    else:
        role_les_code = ""

    role_les_libelle = ""
    if role_LES:
        libelle_les_subject = graph.value(role_LES, rdflib.URIRef("http://exemple/URI7"))
        if libelle_les_subject:
            role_les_libelle = str(libelle_les_subject)

    classification_ucdmo_date_debut = ""
    classification_ucdmo_date_fin = ""
    inscription_value = ""
    ucdMoQuery = graph.query("""
    SELECT ?ucd ?dateDebut ?dateFin ?inscription
    WHERE {
        ?ucd a <http://exemple/URI8> .
        ?ucd <http://exemple/URI9> <""" + str(role_UCD) + """> .
        ?ucd <http://exemple/URI10> <""" + str(role_LES) + """> .
        ?ucd <http://exemple/URI11> ?dateDebut .
        OPTIONAL {?ucd <http://exemple/URI12> ?dateFin .}
        OPTIONAL {?ucd <http://exemple/URI13> ?inscription .}
    }
    """)

    for row in ucdMoQuery:
        classification_ucdmo_date_debut = str(row["dateDebut"])
        classification_ucdmo_date_fin = str(row["dateFin"])
        inscription_value = str(row["inscription"]).split("/")[-1]

    if not classification_ucdmo_date_debut:
        print("Erreur : date debut atih vide ou None")
        return None

    classification_jpucdmo_date_debut = str(date_debut_php1ref) if date_debut_php1ref else ""
    classification_jpucdmo_date_fin = str(date_fin_php1ref) if date_fin_php1ref else ""

    groupe = ""
    groupe_uri = ""
    groupeQuery = graph.query("""
    SELECT DISTINCT ?type
    WHERE {
        ?group <http://exemple/URI14> <""" + str(role_indication_jp) + """> .
        ?group a ?type .
        ?group <http://exemple/URI15> <""" + str(role_UCD) + """> .
        FILTER(regex(str(?type), "JPGroupe" ) )
    }
    """)

    for row in groupeQuery:
        if "Groupe1" in str(row["type"]):
            groupe = "1"
            groupe_uri = "http://exemple/URI16"
        elif "Groupe2" in str(row["type"]):
            groupe = "2"
            groupe_uri = "http://exemple/URI17"
        elif "Groupe3" in str(row["type"]):
            groupe = "3"
            groupe_uri = "http://exemple/URI18"

    rangQuery = graph.query("""
    SELECT ?order
    WHERE {
        ?group a <""" + groupe_uri + """> .
        ?group <http://exemple/URI16> <""" + str(role_UCD) + """> .
        ?group <http://exemple/URI17> <""" + str(role_indication_jp) + """> .
        ?any <http://exemple/URI18> ?group .
        ?any <http://exemple/URI19> <""" + str(role_indication_jp) + """> .
        ?any <http://exemple/URI20> ?order
    }
    """)

    role_assoc_order = ""
    for row in rangQuery:
        role_assoc_order = int(float(str(row["order"])))

    if groupe == "":
        return None

    return {
        "UCD7": role_ucd_code,
        "Libelle medicament": role_ucd_libelle,
        "code indication JP": role_indication_jp_code,
        "code indication LES": role_les_code,
        "libelle indication JP": role_indication_jp_libelle,
        "libelle indication LES": role_les_libelle,
        "date debut indication atih": classification_ucdmo_date_debut,
        "date debut indication php1ref": classification_jpucdmo_date_debut,
        "date fin indication atih": classification_ucdmo_date_fin,
        "date fin indication php1ref": classification_jpucdmo_date_fin,
        "groupe": groupe,
        "rang": role_assoc_order,
        "statut AMM": 0,
        "t2a (oui/non)": inscription_value,
        "statut (T2A, AAP, AAC etc…)": classification_jpucdmo_statut_T2A_ATU,
        "date debut statut T2A": classification_ucdmo_date_debut,
        "date fin statut t2a": date_date_fin_statut_t2a,
    }


def write_csv(nom_fichier_sortie, graph, uris):
    fieldnames = ["fieldnames1", "fieldnames2", "fieldnames3"]
    with open(nom_fichier_sortie, "w", newline="", encoding="windows-1252") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()

        total = len(uris)
        for counter, uri in enumerate(uris, start=1):
            print(f"{counter}/{total}: Processing {uri}")
            row = process_uri(graph, uri)
            if row is not None:
                writer.writerow(row)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Script qui genère un fichier CSV à partir d'un fichier RDF.")
    parser.add_argument("fichier", help="Le nom/chemin du fichier RDF à parser")
    parser.add_argument("nom_du_fichier_de_sortie", help="Le nom du fichier CSV de sortie")
    args = parser.parse_args()

    graph = parse_rdf(args.fichier)
    uris = fetch_uris(graph)
    write_csv(args.nom_du_fichier_de_sortie, graph, uris)

    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()