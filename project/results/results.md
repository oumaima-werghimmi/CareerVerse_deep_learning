# Résultats CareerVerse — test (n = 500)

## Comparaison des modèles

| Modèle | Accuracy | Balanced acc. | Macro-F1 | Top-3 |
|---|---|---|---|---|
| Régression logistique | 0.852 | 0.854 | 0.846 | 0.964 |
| Gradient Boosting | 0.874 | 0.857 | 0.864 | 0.974 |
| MLP | 0.853 ± 0.012 | 0.847 ± 0.015 | 0.846 ± 0.013 | 0.965 ± 0.004 |
| CareerNet (sans labels doux) | 0.871 ± 0.001 | 0.868 ± 0.006 | 0.863 ± 0.002 | 0.971 ± 0.006 |
| CareerNet (+ labels doux) | 0.873 ± 0.008 | 0.873 ± 0.006 | 0.865 ± 0.010 | 0.975 ± 0.006 |

## Par métier (CareerNet (sans labels doux), graine 0)

| Métier | Précision | Rappel | F1 | n |
|---|---|---|---|---|
| Backend Developer | 0.770 | 0.810 | 0.790 | 58 |
| Cloud Engineer | 0.774 | 0.788 | 0.781 | 52 |
| Cybersecurity Analyst | 0.911 | 0.927 | 0.919 | 55 |
| Data Scientist | 0.956 | 0.956 | 0.956 | 68 |
| Database Administrator | 0.889 | 0.906 | 0.897 | 53 |
| DevOps Engineer | 0.857 | 0.750 | 0.800 | 24 |
| Frontend/Web Developer | 0.853 | 0.889 | 0.871 | 72 |
| QA/Test Engineer | 0.873 | 0.842 | 0.857 | 57 |
| System/Network Admin | 0.947 | 0.885 | 0.915 | 61 |

## Robustesse (macro-F1 selon le nombre max de labs faits)

| Modèle | 0 | 1 | 2 | 3 | 4 | 5 | tous |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 0.606 | 0.641 | 0.669 | 0.721 | 0.772 | 0.821 | 0.864 |
| MLP | 0.641 | 0.693 | 0.716 | 0.766 | 0.814 | 0.812 | 0.834 |
| CareerNet (sans labels doux) | 0.624 | 0.664 | 0.720 | 0.773 | 0.812 | 0.833 | 0.865 |

## Importance par permutation (baisse du macro-F1)

| Groupe | Baisse |
|---|---|
| Career Labs (tous) | 0.503 |
| Auto-évaluation | 0.220 |
| Lab network_config | 0.084 |
| Lab devops_pipeline | 0.080 |
| Lab web_ui | 0.074 |
| Lab qa_testing | 0.065 |
| Lab security_ctf | 0.064 |
| Lab cloud_deploy | 0.063 |
| Lab sql_database | 0.063 |
| Lab backend_api | 0.061 |
| Lab data_analysis | 0.049 |
| Intérêts déclarés | 0.028 |
| Notes académiques | 0.028 |

## Équité (genre non utilisé en entrée)

| Genre | n | Accuracy | Macro-F1 |
|---|---|---|---|
| F | 231 | 0.887 | 0.882 |
| M | 269 | 0.859 | 0.852 |

Écart d'accuracy F - M : +0.029, IC 95 % bootstrap [-0.0296, 0.0863] -> contient 0 : pas d'écart significatif
