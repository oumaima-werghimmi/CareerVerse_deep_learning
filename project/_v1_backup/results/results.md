# Résultats CareerVerse — test (n = 500)

## Comparaison des modèles

| Modèle | Accuracy | Balanced acc. | Macro-F1 | Top-3 |
|---|---|---|---|---|
| Régression logistique | 0.896 | 0.895 | 0.888 | 0.988 |
| Gradient Boosting | 0.892 | 0.871 | 0.875 | 0.986 |
| MLP | 0.898 ± 0.005 | 0.891 ± 0.008 | 0.890 ± 0.006 | 0.982 ± 0.003 |
| CareerNet (sans labels doux) | 0.909 ± 0.003 | 0.909 ± 0.003 | 0.903 ± 0.007 | 0.989 ± 0.002 |
| CareerNet (+ labels doux) | 0.905 ± 0.013 | 0.901 ± 0.011 | 0.896 ± 0.014 | 0.995 ± 0.001 |

## Par métier (CareerNet (sans labels doux), graine 0)

| Métier | Précision | Rappel | F1 | n |
|---|---|---|---|---|
| Backend Developer | 0.891 | 0.845 | 0.867 | 58 |
| Cloud Engineer | 0.870 | 0.769 | 0.816 | 52 |
| Cybersecurity Analyst | 0.879 | 0.927 | 0.903 | 55 |
| Data Scientist | 0.971 | 0.971 | 0.971 | 68 |
| Database Administrator | 0.980 | 0.906 | 0.941 | 53 |
| DevOps Engineer | 0.852 | 0.958 | 0.902 | 24 |
| Frontend/Web Developer | 0.946 | 0.972 | 0.959 | 72 |
| QA/Test Engineer | 0.912 | 0.912 | 0.912 | 57 |
| System/Network Admin | 0.848 | 0.918 | 0.882 | 61 |

## Robustesse (macro-F1 selon le nombre max de labs faits)

| Modèle | 0 | 1 | 2 | 3 | 4 | 5 | tous |
|---|---|---|---|---|---|---|---|
| Gradient Boosting | 0.736 | 0.745 | 0.768 | 0.781 | 0.812 | 0.842 | 0.875 |
| MLP | 0.773 | 0.809 | 0.798 | 0.823 | 0.836 | 0.858 | 0.886 |
| CareerNet (sans labels doux) | 0.734 | 0.771 | 0.791 | 0.812 | 0.842 | 0.884 | 0.906 |

## Importance par permutation (baisse du macro-F1)

| Groupe | Baisse |
|---|---|
| Auto-évaluation | 0.481 |
| Career Labs (tous) | 0.249 |
| Lab cloud_deploy | 0.062 |
| Intérêts déclarés | 0.051 |
| Lab devops_pipeline | 0.050 |
| Lab network_config | 0.045 |
| Lab backend_api | 0.040 |
| Lab web_ui | 0.037 |
| Lab sql_database | 0.032 |
| Lab qa_testing | 0.030 |
| Lab security_ctf | 0.021 |
| Notes académiques | 0.010 |
| Lab data_analysis | 0.007 |

## Équité (genre non utilisé en entrée)

| Genre | n | Accuracy | Macro-F1 |
|---|---|---|---|
| F | 231 | 0.905 | 0.898 |
| M | 269 | 0.914 | 0.911 |

Écart d'accuracy F - M : -0.010, IC 95 % bootstrap [-0.0611, 0.0403] -> contient 0 : pas d'écart significatif
