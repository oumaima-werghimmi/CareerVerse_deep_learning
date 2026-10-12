# CareerVerse : collecte et construction du dataset

## Pipeline

```
step1_collect_onet.py       O*NET (public)  →  data/career_profiles.csv   (9 métiers × 13 compétences, 0-10)
step2_generate_students.py  profils O*NET   →  data/careerverse_students.csv (étudiants + Career Labs + label)
(plus tard)                 vrais utilisateurs CareerVerse → même schéma de colonnes
```

```bash
pip install pandas numpy requests scikit-learn
python step1_collect_onet.py                 # télécharge O*NET (ou --zip fichier.zip)
python step2_generate_students.py --n 5000   # génère le dataset
```

Si tu lances l'étape 2 sans l'étape 1, elle utilise des profils « de secours » définis à la main (colonne `profile_source = fallback`). Pour le rendu final, lance d'abord l'étape 1 pour que les profils viennent d'O*NET (`profile_source = onet`).

## Comment le label est défini (le point à défendre devant le jury)

`career_label` **n'est pas** « le lab où l'étudiant a le meilleur score ». La simulation fonctionne ainsi :

1. Chaque étudiant reçoit un métier latent : c'est la vérité terrain.
2. Ses aptitudes réelles sont tirées autour du profil O*NET de ce métier, plus un niveau général et du bruit. Les classes se chevauchent donc.
3. La plateforme **observe** seulement des signaux bruités : auto-évaluation biaisée, notes, intérêts déclarés (20 % sont trompeurs) et Career Labs (score, temps, tentatives, indices, plaisir). Les labs non faits sont des valeurs manquantes (environ 41 %).
4. Le modèle doit retrouver le métier à partir de ces observations. Le fichier `*_latent.csv` sert uniquement à l'analyse, jamais à l'entraînement.

Il existe aussi un label doux, `fit_<métier>`, qui donne une probabilité d'adéquation à chaque métier. Il est utile pour entraîner un réseau avec la KL-divergence ou pour faire une recommandation top-3.

## Vérification (5 000 étudiants, découpage 80/20)

| Features utilisées | Précision |
|---|---|
| Toutes (régression logistique) | 0,886 |
| Toutes (gradient boosting) | 0,892 |
| Career Labs seulement | 0,81 |
| Auto-évaluation seulement | 0,71 |
| Intérêts seulement | 0,38 |

Le jeu est apprenable sans être trivial, et les Career Labs portent le plus d'information. C'est un bon argument pour la plateforme. Pour rendre la tâche plus difficile, augmente le bruit dans `generate()`.

## Colonnes à exclure des features
`student_id`, `profile_source`, `gender` (à garder seulement pour l'audit d'équité), `career_label` et `fit_*` (ce sont des labels). Voir `data/data_dictionary.csv`.

## Autres sources (optionnelles)
- **ESCO** (API gratuite) : sert à traduire et aligner les compétences en français ou avec la nomenclature européenne.
- **Kaggle, « Computer Science Students Career Prediction »** : sert de *validation externe* sur de vraies réponses d'étudiants, après avoir mis ses colonnes en correspondance avec les `self_*` et `interest_*`.
- **Vrais utilisateurs CareerVerse** : collecter exactement les mêmes colonnes. Le label réel vient ensuite du choix final de l'étudiant, validé par un mentor ou par sa satisfaction à 6 mois. On fait un pré-entraînement sur les données simulées puis un fine-tuning sur les données réelles.
