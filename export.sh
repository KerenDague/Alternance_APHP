#!/bin/bash
# Description du script : Ce script va lancer les exports (via run_export.sh) puis modifier l'encodage des fichiers .csv. Ensuite il met les fichiers .csv contenant une phrase d'erreur type dans un dossier "poubelle" et renomme les autres correctement.

# Se placer dans le repertoire
if ! cd /exemple/de/chemin; then
    echo "Impossible de se placer dans le rpertoire." >&2
    exit 1
fi

# Supprimer et recreer le dossier de résultats
rm -rf result
mkdir result
# Supprimer et recreer le dossier 'poubelle'
rm -rf poubelle
mkdir poubelle
fichiers_vides_trouves=false

# Lancement des exports
./run_export.sh

# Encodage de tous les fichiers de UTF-8 sans BOM vers UTF-8 avec BOM
cd result
sed -i '1s/^\(\xef\xbb\xbf\)\?/\xef\xbb\xbf/' report-RefBio_*

# Debut de l'export et vérification des fichiers
for file in *.csv; do
    if [ ! -e "$file" ]; then
        echo "Le fichier '$file' n'existe pas."
        continue
	fi

    # Verifier si le fichier contient l'erreur
    if grep -q "The initial request for this report has brought no result" "$file"; then
        mv "$file" "../poubelle/$file"
        fichiers_vides_trouves=true
    else
		# Renommer tous les fichiers 
		if [[ "$file" == "exemple_nom_de_fichier1.csv" ]]; then
		mv exemple_nouveau_nom_de_fichier1.$(date +"%Y%m%d").csv
		elif [[ "$file" == "exemple_nom_de_fichier2.csv" ]]; then
		mv exemple_nouveau_nom_de_fichier2.$(date +"%Y%m%d").csv
		elif [[ "$file" == "exemple_nom_de_fichier3.csv" ]]; then
		mv exemple_nouveau_nom_de_fichier3.$(date +"%Y%m%d").csv
        fi
    fi
done

# Verification finale
if [ "$fichiers_vides_trouves" = "false" ]; then
    echo "Aucun fichier vide trouvé (contenant la phrase d'erreur)."
fi

# Transfert des fichiers sur le répertoire de sortie
cd /exemple/de/chemin
cp *.csv /exemple/de/chemin/de/repertoire/de/sortie
