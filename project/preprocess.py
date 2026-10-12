
"""
CareerVerse — Étape 3 : contrôle qualité + prétraitement
========================================================

Entrée :
    data/careerverse_students.csv
    # Fichier CSV qui contient les données des étudiants.

Sorties :
    artifacts/dataset.npz
    # Contient les données numériques séparées en train, validation et test.

    artifacts/preprocess.json
    # Contient les paramètres du prétraitement, comme les moyennes,
    # les écarts-types et les codes des carrières.

Objectif :
    # Vérifier la qualité des données, les préparer et les enregistrer
    # dans un format utilisable par un modèle de Deep Learning.

Choix de prétraitement :
    1. Exclure les identifiants, les labels et les variables sensibles.
    2. Séparer les informations statiques des résultats des laboratoires.
    3. Transformer les nombres de tentatives et d'indices avec log1p.
    4. Normaliser les données avec les statistiques du train uniquement.
    5. Séparer les données en train (4000), validation (500) et test (500).
    6. Calculer des poids pour compenser le déséquilibre des carrières.
"""

# Importer json pour enregistrer les paramètres au format JSON.
import json

# Importer os pour créer les dossiers et construire les chemins des fichiers.
import os

# Importer NumPy pour manipuler les tableaux numériques.
import numpy as np

# Importer Pandas pour lire et analyser le fichier CSV.
import pandas as pd

# Importer la fonction qui permet de séparer les données en plusieurs groupes.
from sklearn.model_selection import train_test_split


# Chemin du fichier CSV contenant les données originales des étudiants.
DATA = "data/careerverse_students.csv"

# Dossier dans lequel seront enregistrés les résultats du prétraitement.
OUT = "artifacts"

# Graine aléatoire fixe pour pouvoir reproduire les mêmes séparations.
SEED = 42


# Liste des neuf laboratoires du projet CareerVerse.
LABS = [
    "lab_backend_api",
    "lab_web_ui",
    "lab_devops_pipeline",
    "lab_cloud_deploy",
    "lab_security_ctf",
    "lab_data_analysis",
    "lab_sql_database",
    "lab_network_config",
    "lab_qa_testing"
]

# Liste des cinq mesures conservées pour chaque laboratoire.
LAB_METRICS = [
    "score",          # Score obtenu au laboratoire.
    "time_min",       # Temps passé, en minutes.
    "attempts",       # Nombre de tentatives.
    "hints_used",     # Nombre d'indices utilisés.
    "enjoyment_1_5"   # Niveau de plaisir ressenti, de 1 à 5.
]

# Mesures de comptage qui nécessitent une transformation logarithmique.
LOG_METRICS = {"attempts", "hints_used"}

# Colonnes exclues des entrées du modèle.
EXCLUDE = {
    "student_id",     # Identifiant : ne décrit pas les compétences.
    "profile_source", # Métadonnée constante : n'apporte pas d'information.
    "gender",         # Variable sensible, conservée pour l'audit d'équité.
    "career_label"    # Réponse que le modèle doit apprendre à prédire.
}


def quality_report(df):
    # Définir une fonction qui vérifie la qualité du dataset.
    # df représente le tableau contenant les données des étudiants.

    # Afficher un titre pour organiser les résultats dans la console.
    print("=" * 60, "\nCONTRÔLE QUALITÉ\n" + "=" * 60)

    # Afficher le nombre de lignes et de colonnes du dataset.
    # len(df) donne le nombre de lignes.
    # df.shape[1] donne le nombre de colonnes.
    print(f"lignes={len(df)}  colonnes={df.shape[1]}")

    # Compter les identifiants student_id qui apparaissent plusieurs fois.
    # duplicated() marque les répétitions et sum() les compte.
    print(f"doublons student_id : {df.student_id.duplicated().sum()}")

    # Retirer temporairement student_id et compter les lignes identiques.
    # Cela permet de détecter des lignes identiques même si leurs IDs diffèrent.
    print(
        f"lignes entièrement dupliquées : "
        f"{df.drop(columns='student_id').duplicated().sum()}"
    )

    # Définir des expressions régulières pour sélectionner des colonnes
    # et leurs valeurs minimale et maximale autorisées.
    checks = {
        r"self_": (0, 10),             # Autoévaluations entre 0 et 10.
        r"interest_": (1, 5),          # Intérêts entre 1 et 5.
        r"_20$": (0, 20),              # Notes sur 20.
        r"__score$": (0, 100),         # Scores des laboratoires sur 100.
        r"__enjoyment_1_5$": (1, 5)    # Plaisir ressenti entre 1 et 5.
    }

    # Initialiser le compteur de valeurs hors limites.
    bad = 0

    # Parcourir chaque règle de vérification.
    # pat représente le motif de nom de colonne.
    # lo et hi représentent les limites minimale et maximale.
    for pat, (lo, hi) in checks.items():

        # Sélectionner les colonnes dont les noms correspondent au motif.
        cols = df.filter(regex=pat).columns

        # Compter les valeurs inférieures à lo ou supérieures à hi.
        # sum().sum() additionne les erreurs de toutes les colonnes.
        n = ((df[cols] < lo) | (df[cols] > hi)).sum().sum()

        # Ajouter le nombre d'erreurs au compteur global.
        bad += n

        # Afficher un avertissement uniquement si des erreurs existent.
        if n:
            print(f"   !! {n} valeurs hors [{lo},{hi}] pour {pat}")

    # Afficher le nombre total de valeurs hors limites détectées.
    print(f"valeurs hors bornes : {bad}")

    # Sélectionner les colonnes hors laboratoires.
    # "__" permet ici de repérer les colonnes des mesures des laboratoires.
    # Les colonnes fit_* sont également exclues de ce contrôle.
    non_lab = [
        c for c in df.columns
        if "__" not in c and not c.startswith("fit_")
    ]

    # Compter les valeurs manquantes dans les colonnes sélectionnées.
    # isna() détecte les valeurs manquantes.
    print(
        f"manquants hors labs : "
        f"{int(df[non_lab].isna().sum().sum())}"
    )

    # Calculer la proportion moyenne de scores de laboratoire manquants.
    # Un score manquant est considéré comme un laboratoire non réalisé.
    print(
        f"labs non faits : "
        f"{df.filter(like='__score').isna().mean().mean():.1%} en moyenne"
    )

    # Compter les étudiants qui n'ont réalisé aucun laboratoire.
    # n_labs_done indique le nombre de laboratoires réalisés.
    print(
        f"étudiants sans aucun lab : "
        f"{(df.n_labs_done == 0).sum()}"
    )

    # Vérifier la cohérence entre completed et le score de chaque laboratoire.
    # La règle supposée est : completed = 1 si score >= 50, sinon 0.
    # Cette règle doit correspondre à la façon dont les données ont été générées.
    incoh = sum(
        (
            (df[f"{l}__score"] >= 50).astype(float)
            != df[f"{l}__completed"]
        )[df[f"{l}__score"].notna()].sum()
        for l in LABS
    )

    # Afficher le nombre de différences entre completed et la règle du score.
    print(
        f"incohérences completed/score : {int(incoh)} "
        f"(completed est calculé avant l'arrondi du score à 0,1 : un score de 49,95-49,99 "
        "est affiché 50.0 avec completed=0) -> colonne redondante, non utilisée en entrée"
    )

    # Compter le nombre d'étudiants pour chaque carrière.
    vc = df.career_label.value_counts()

    # Afficher le rapport entre la classe la plus représentée
    # et celle qui contient le moins d'étudiants.
    print(
        f"\nrépartition des classes "
        f"(ratio max/min = {vc.max() / vc.min():.1f}) :"
    )

    # Afficher le nombre d'étudiants par carrière.
    print(vc.to_string())

    # Sélectionner les colonnes des scores de compatibilité fit_*.
    fit = df.filter(like="fit_")

    # Vérifier si la somme des scores fit_* est proche de 1 pour chaque étudiant.
    # min() affiche la somme minimale et max() la somme maximale.
    print(
        f"\nsomme des labels doux fit_* : "
        f"min={fit.sum(axis=1).min():.3f} "
        f"max={fit.sum(axis=1).max():.3f}"
    )

    # Afficher une ligne de séparation pour terminer le rapport.
    print("=" * 60)


def main(data=DATA, out=OUT):
    # Définir la fonction principale qui exécute tout le prétraitement.

    # Créer le dossier artifacts s'il n'existe pas déjà.
    # exist_ok=True évite une erreur si le dossier existe.
    os.makedirs(out, exist_ok=True)

    # Lire le fichier CSV et stocker son contenu dans le tableau df.
    df = pd.read_csv(data)

    # Exécuter les contrôles de qualité avant de préparer les données.
    quality_report(df)

    # Récupérer les différentes carrières présentes dans career_label.
    # unique() supprime les répétitions et sorted() trie les noms.
    classes = sorted(df.career_label.unique())

    # Associer un nombre à chaque carrière.
    # enumerate() produit les couples (indice, carrière).
    # map() remplace le nom de la carrière par son indice.
    # values récupère le tableau des valeurs.
    y = df.career_label.map({
        c: i for i, c in enumerate(classes)
    }).values

    # Construire les noms des colonnes de compatibilité fit_*.
    # Cette instruction suppose que leurs suffixes correspondent
    # exactement aux noms présents dans classes.
    fit_cols = [f"fit_{c}" for c in classes]

    # Récupérer les neuf scores de compatibilité pour chaque étudiant.
    # astype(np.float32) convertit les valeurs en nombres flottants
    # de précision 32 bits, adaptés à de nombreux calculs de Deep Learning.
    soft = df[fit_cols].values.astype(np.float32)

    # Sélectionner les entrées statiques du modèle.
    # Exclure les colonnes de EXCLUDE.
    # Exclure les colonnes contenant "__", utilisées pour les laboratoires.
    # Exclure les colonnes fit_* car elles représentent des cibles secondaires.
    static_cols = [
        c for c in df.columns
        if c not in EXCLUDE
        and "__" not in c
        and not c.startswith("fit_")
    ]

    # Construire le tableau des laboratoires.
    # Pour chaque laboratoire, récupérer les cinq mesures.
    # Le premier np.stack rassemble les mesures d'un laboratoire.
    # Le deuxième np.stack rassemble les neuf laboratoires.
    # Le résultat possède la forme (nombre d'étudiants, 9, 5).
    # astype(float) convertit les données en nombres flottants.
    lab = np.stack([
        np.stack([
            df[f"{l}__{m}"].values
            for m in LAB_METRICS
        ], -1)
        for l in LABS
    ], 1).astype(float)

    # Construire le masque des laboratoires réalisés.
    # lab[:, :, 0] récupère les scores de tous les laboratoires.
    # np.isnan() détecte les scores manquants.
    # ~ inverse le résultat : True indique un score présent.
    mask = ~np.isnan(lab[:, :, 0])

    # Parcourir les cinq mesures des laboratoires.
    for j, m in enumerate(LAB_METRICS):

        # Appliquer log1p uniquement aux mesures définies dans LOG_METRICS.
        if m in LOG_METRICS:

            # Transformer chaque valeur x en log(1 + x).
            # Cette opération réduit l'asymétrie des comptages.
            lab[:, :, j] = np.log1p(lab[:, :, j])

    # Extraire les données statiques sélectionnées précédemment.
    # Le tableau obtenu contient une ligne par étudiant.
    static = df[static_cols].values.astype(float)

    # Créer un tableau contenant les indices de tous les étudiants.
    # Par exemple, pour 5000 étudiants : 0, 1, 2, ..., 4999.
    idx = np.arange(len(df))

    # Séparer les indices en deux groupes :
    # 4000 étudiants pour le train et 1000 pour validation + test.
    # Un entier dans test_size fixe un nombre exact d'exemples.
    # stratify=y conserve approximativement la proportion des carrières.
    # random_state=SEED permet de reproduire la séparation.
    tr, tmp = train_test_split(
        idx,
        test_size=1000,
        stratify=y,
        random_state=SEED
    )

    # Séparer les 1000 restants en deux groupes de même taille.
    # 500 deviennent validation et 500 deviennent test.
    # stratify=y[tmp] préserve la répartition des classes dans ce groupe.
    va, te = train_test_split(
        tmp,
        test_size=0.50,
        stratify=y[tmp],
        random_state=SEED
    )

    # Calculer la moyenne de chaque colonne statique sur le train seulement.
    s_mean = static[tr].mean(0)

    # Calculer l'écart-type de chaque colonne statique sur le train.
    # 1e-6 est ajouté pour éviter une division par zéro.
    s_std = static[tr].std(0) + 1e-6

    # Sélectionner les mesures des laboratoires réalisés par les étudiants
    # du train uniquement.
    # mask[tr] indique quels laboratoires ont été réalisés.
    lab_tr = lab[tr][mask[tr]]

    # Calculer la moyenne de chacune des cinq mesures sur les labs réalisés.
    l_mean = lab_tr.mean(0)

    # Calculer l'écart-type des cinq mesures sur les labs réalisés.
    # Ajouter une petite valeur pour éviter une division par zéro.
    l_std = lab_tr.std(0) + 1e-6

    # Normaliser toutes les données statiques avec les statistiques du train.
    # La moyenne est soustraite et le résultat est divisé par l'écart-type.
    static_n = (static - s_mean) / s_std

    # Normaliser les mesures des laboratoires réalisés.
    # mask[..., None] ajoute une dimension au masque pour comparer
    # correctement les cinq mesures de chaque laboratoire.
    # Les laboratoires non réalisés reçoivent des zéros après normalisation.
    lab_n = np.where(
        mask[..., None],
        (lab - l_mean) / l_std,
        0.0
    )

    # Compter les exemples de chaque classe dans le train.
    # minlength garantit une position pour chaque carrière.
    counts = np.bincount(y[tr], minlength=len(classes))

    # Calculer un poids inversement proportionnel à la fréquence de chaque classe.
    # Une classe moins représentée reçoit un poids plus élevé.
    # Ces poids devront être utilisés par le code d'entraînement.
    class_w = len(tr) / (len(classes) * counts)

    # Créer un dictionnaire vide pour stocker les tableaux à enregistrer.
    arrays = {}

    # Parcourir les trois groupes de données.
    # name vaut train, val ou test.
    # ids contient les indices des étudiants du groupe correspondant.
    for name, ids in [
        ("train", tr),
        ("val", va),
        ("test", te)
    ]:

        # Enregistrer les caractéristiques statiques normalisées.
        # float32 permet de réduire la mémoire utilisée.
        arrays[f"{name}_static"] = static_n[ids].astype(np.float32)

        # Enregistrer les mesures des laboratoires normalisées.
        arrays[f"{name}_lab"] = lab_n[ids].astype(np.float32)

        # Enregistrer le masque qui identifie les laboratoires réalisés.
        arrays[f"{name}_mask"] = mask[ids]

        # Enregistrer les classes cibles sous forme de nombres.
        arrays[f"{name}_y"] = y[ids]

        # Enregistrer les neuf scores de compatibilité par étudiant.
        arrays[f"{name}_soft"] = soft[ids]

        # Conserver le genre séparément pour les audits d'équité.
        # U8 indique une chaîne de caractères Unicode pouvant contenir
        # jusqu'à huit caractères.
        arrays[f"{name}_gender"] = np.asarray(
            df.gender.values[ids],
            dtype="U8"
        )

        # Conserver les identifiants pour retrouver les étudiants
        # lors des analyses, sans les fournir comme entrées au modèle.
        # U16 indique une chaîne Unicode de longueur maximale 16.
        arrays[f"{name}_id"] = np.asarray(
            df.student_id.values[ids],
            dtype="U16"
        )

    # Enregistrer tous les tableaux dans un fichier NPZ compressé.
    # **arrays transmet chaque entrée du dictionnaire comme un tableau nommé.
    np.savez_compressed(
        os.path.join(out, "dataset.npz"),
        **arrays
    )

    # Préparer les paramètres nécessaires pour reproduire le prétraitement.
    meta = {
        # Liste des carrières et ordre de leur encodage numérique.
        "classes": classes,

        # Liste des colonnes statiques utilisées comme entrées.
        "static_cols": static_cols,

        # Liste des neuf laboratoires.
        "labs": LABS,

        # Liste des cinq mesures conservées pour chaque laboratoire.
        "lab_metrics": LAB_METRICS,

        # Liste des mesures auxquelles log1p a été appliqué.
        "log_metrics": sorted(LOG_METRICS),

        # Convertir les moyennes NumPy en listes compatibles avec JSON.
        "static_mean": s_mean.tolist(),

        # Convertir les écarts-types statiques en listes.
        "static_std": s_std.tolist(),

        # Enregistrer les moyennes des mesures des laboratoires.
        "lab_mean": l_mean.tolist(),

        # Enregistrer les écarts-types des mesures des laboratoires.
        "lab_std": l_std.tolist(),

        # Enregistrer les poids calculés pour chaque classe.
        "class_weights": class_w.tolist(),

        # Enregistrer le nombre d'étudiants dans chaque groupe.
        "split_sizes": {
            "train": len(tr),
            "val": len(va),
            "test": len(te)
        },

        # Enregistrer la graine aléatoire utilisée.
        "seed": SEED
    }

    # Écrire les paramètres dans le fichier preprocess.json.
    # indent=2 rend le fichier plus facile à lire.
    # ensure_ascii=False conserve les caractères non ASCII.
    json.dump(
        meta,
        open(os.path.join(out, "preprocess.json"), "w", encoding="utf-8"),
        indent=2,
        ensure_ascii=False
    )

    # Afficher le nombre d'étudiants dans chaque groupe
    # et la forme des données préparées.
    print(
        f"\n[OK] train={len(tr)} val={len(va)} test={len(te)} | "
        f"{len(static_cols)} features statiques + "
        f"{len(LABS)} labs x {len(LAB_METRICS)} mesures"
    )

    # Afficher le poids calculé pour chaque carrière.
    # zip associe chaque nom de carrière à son poids.
    # round(..., 2) arrondit le poids à deux décimales.
    print(
        "poids de classes :",
        {
            c: round(float(w), 2)
            for c, w in zip(classes, class_w)
        }
    )


# Vérifier que ce fichier est exécuté directement.
# Si ce fichier est importé depuis un autre script, main() ne sera pas
# automatiquement exécutée par cette condition.
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    # Lancer toutes les étapes du contrôle qualité et du prétraitement.
    main(args.data, args.out)
