# CareerVerse : module de recommandation de métier (deep learning)

## 1. Vérification de tes données : ce qui a été corrigé

| Problème trouvé dans `data.zip` | Cause | Correction |
|---|---|---|
| **Data Scientist à 10/10 partout** (programmation, Linux, Docker, Cloud…) | O*NET ne liste que 87 technologies pour ce métier, contre 250 à 430 pour les autres. La *proportion* de mots-clés trouvés est donc gonflée. | Le score technologique mesure maintenant l'**emphase relative** au sein du métier, puis il est normalisé entre les métiers. |
| **Data Scientist vide** en `problem_solving` et `systems_thinking`, puis rempli silencieusement par 5 | O*NET 30.1 n'a aucune donnée Skills/Knowledge/Abilities pour 15-2051.00. | On utilise des métiers proches à la place : Statisticians (15-2041) et Computer & Info Research Scientists (15-1221). L'étape 2 **s'arrête** désormais au lieu de remplir par une valeur arbitraire. |
| Profils O*NET encore bruités (ex. Frontend = 10 en Linux) | La correspondance par mots-clés est trop grossière pour distinguer finement des métiers IT. | Profil final = **50 % O*NET + 50 % profil expert**, avec un contrôle automatique : chaque métier doit être 1er ou 2e sur sa compétence signature. Les 9 métiers passent. |
| `lab_*__completed` | Redondant avec `score ≥ 50` (18 écarts dus à l'arrondi). | Colonne supprimée des entrées. |
| Fichier vide `data/python` | Création accidentelle. | À supprimer. |

**Effet :** le label doux `fit_*` concorde avec `career_label` pour **77 %** des étudiants, contre **60 %** avant. Les anciens profils rendaient le dataset en partie contradictoire.

Contrôle qualité sur le dataset corrigé : 0 doublon, 0 valeur hors bornes, 0 manquant hors labs, 41 % de labs non faits (c'est normal, c'est le fonctionnement de la plateforme), et 3 fois plus d'étudiants dans la classe la plus fréquente que dans la plus rare (DevOps).

## 2. Pipeline

```bash
pip install pandas numpy requests scikit-learn torch matplotlib
python step1_collect_onet.py --zip ../data/onet_raw/db_30_1_text.zip   # profils métiers (corrigés)
python step2_generate_students.py --n 5000 --seed 42                # étudiants + Career Labs
python preprocess.py        # contrôle qualité, normalisation, split 70/15/15 -> artifacts/
python train.py             # entraîne CareerNet -> artifacts/careernet.pt   (~1 min sur CPU)
python experiments.py       # comparaison des modèles + analyses + figures -> results/  (~5 min)
python predict.py example_student.json   # top-3 métiers pour un étudiant
```

## 3. Prétraitement (`preprocess.py`)

- **Entrées exclues :** `student_id`, `profile_source`, `career_label`, `fit_*` (ce sont des cibles), `gender` (attribut sensible, gardé seulement pour l'audit d'équité) et `completed` (redondant).
- **24 variables statiques :** année, 4 notes, 13 auto-évaluations, 5 intérêts, et le nombre de labs faits.
- **Labs :** un tenseur `(9 labs × 5 mesures)` (score, temps, tentatives, indices, plaisir), accompagné d'un **masque**. Un lab non fait n'est jamais remplacé par une fausse valeur.
- `log1p` sur les tentatives et les indices, puis standardisation calculée **sur le train seulement**, pour éviter toute fuite d'information.
- Découpage stratifié 70/15/15 (graine 42) et poids de classes pour compenser le déséquilibre (DevOps a un poids de 2,31).

## 4. Modèle : CareerNet (`model.py`)

```
profil (24) ─► MLP 24→128→64 ────────────────────────────┐
                                                          ├─► concat ─► MLP ─► 9 métiers (softmax)
labs (9×5) ─► Linear + embedding du lab ─► [CLS]+9 tokens ─► Transformer (2 couches, 4 têtes, masque) ─┘
```

Pourquoi cette architecture :
- Chaque étudiant fait un sous-ensemble différent de labs. Le **masque d'attention** ignore les labs absents.
- L'attention apprend les **interactions entre labs** (par exemple, CTF + réseau oriente vers la cybersécurité).
- Ajouter un lab revient à ajouter un token : l'architecture s'étend facilement.

Entraînement : entropie croisée pondérée avec label smoothing de 0,05, AdamW (lr 1e-3, weight decay 1e-4), scheduler cosinus avec warm-up, gradient clipping et **lab dropout de 15 %** (on masque des labs au hasard pour simuler des profils incomplets). L'early stopping se fait sur le macro-F1 de validation. Le modèle compte 88 k paramètres.

Remarque sur les courbes : la perte d'entraînement est plus haute que la perte de validation. C'est normal, car dropout, label smoothing et lab dropout ne s'appliquent qu'à l'entraînement.

## 5. Résultats (test, n = 750, 3 graines pour les modèles deep learning)

Voir `results/results.md` et `results/*.png`.

| Modèle | Accuracy | Macro-F1 | Top-3 |
|---|---|---|---|
| Régression logistique | 0,889 | 0,878 | 0,980 |
| Gradient Boosting | 0,889 | 0,878 | 0,983 |
| MLP | 0,898 ± 0,002 | 0,889 ± 0,003 | 0,985 |
| **CareerNet (retenu)** | **0,904 ± 0,004** | **0,895 ± 0,006** | 0,985 |
| CareerNet + labels doux | 0,902 ± 0,004 | 0,892 ± 0,005 | 0,988 |

Lecture honnête de ces résultats :
- CareerNet est le meilleur modèle, mais l'avance est **modeste** : +1,7 point de macro-F1 par rapport aux baselines classiques et +0,6 point par rapport au MLP. Le plafond vient du bruit volontairement mis dans les données.
- Les labels doux n'apportent **aucun gain significatif**. On garde donc la version sans labels doux, la seule utilisable sur de vraies données.
- **Top-3 à environ 98,5 %** : en pratique, afficher 3 métiers recommandés est très fiable.
- **Point faible : DevOps** (F1 ≈ 0,80). C'est la classe la plus rare, et elle est confondue avec Cloud.
- **Robustesse :** sans aucun lab, le macro-F1 est d'environ 0,78, contre 0,90 avec tous les labs. Les Career Labs apportent donc environ **12 points**, ce qui est l'argument principal pour la plateforme. Quand l'étudiant a fait moins de 5 labs, le MLP est légèrement meilleur. Une piste d'amélioration est d'augmenter le lab dropout.
- **Importance :** l'auto-évaluation domine (0,50), devant les labs (0,19). ⚠ C'est un biais possible de la simulation, où l'auto-évaluation est une copie bruitée des vraies aptitudes. Dans la réalité, l'auto-évaluation est souvent moins fiable, et les labs devraient compter davantage.
- **Équité :** écart d'accuracy F − M de −1,8 point, avec un intervalle de confiance à 95 % de [−5,9 ; +2,3]. Il **n'est pas significatif**. Cet audit est à refaire sur les vraies données.

## 6. Limites à écrire dans le rapport

1. Les données sont **simulées**. Les performances mesurent la capacité du modèle à retrouver le processus de simulation, pas encore une validité réelle.
2. Les profils métiers viennent pour moitié d'un jugement d'expert.
3. Pour passer au réel : collecter les mêmes colonnes sur la plateforme. Le label vient du choix final de l'étudiant, validé par un mentor. On pré-entraîne sur la simulation, puis on fait un fine-tuning sur les données réelles, avec `python train.py --soft_weight 0`.
