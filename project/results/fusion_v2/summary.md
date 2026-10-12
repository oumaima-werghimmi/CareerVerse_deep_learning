# Fusion profil / labs — dataset v2 (test n = 500, graines [0, 1, 2])

## Performances sur données d'origine (test)

| Config | Accuracy | Balanced acc. | Macro-F1 | Top-3 | NLL | ECE | ECE après T |
|---|---|---|---|---|---|---|---|
| A. Profil seul | 0.681 ± 0.004 | 0.671 ± 0.006 | 0.660 ± 0.006 | 0.921 ± 0.002 | 0.930 ± 0.005 | 0.048 ± 0.007 | 0.040 ± 0.002 |
| B. Labs seuls | 0.829 ± 0.013 | 0.828 ± 0.014 | 0.821 ± 0.013 | 0.961 ± 0.006 | 0.541 ± 0.025 | 0.045 ± 0.012 | 0.031 ± 0.010 |
| C. CareerNet | 0.871 ± 0.001 | 0.868 ± 0.006 | 0.863 ± 0.002 | 0.971 ± 0.006 | 0.425 ± 0.009 | 0.041 ± 0.003 | 0.032 ± 0.007 |
| D. CareerNet à porte | 0.870 ± 0.007 | 0.868 ± 0.004 | 0.861 ± 0.007 | 0.968 ± 0.003 | 0.434 ± 0.010 | 0.045 ± 0.004 | 0.039 ± 0.002 |
| C+. CareerNet + labels doux | 0.873 ± 0.008 | 0.873 ± 0.006 | 0.865 ± 0.010 | 0.975 ± 0.006 | 0.478 ± 0.026 | 0.105 ± 0.016 | 0.040 ± 0.010 |
| E. CareerNet sans plaisir | 0.814 ± 0.009 | 0.801 ± 0.017 | 0.799 ± 0.014 | 0.959 ± 0.002 | 0.566 ± 0.022 | 0.044 ± 0.010 | 0.045 ± 0.009 |

## Robustesse (test, macro-F1 ; entre parenthèses : % d'étudiants dont le métier n°1 change)

| Config | self +2 | self −2 | bruit σ=2 | inflation autre métier | 0 lab | 1 lab | 2 labs | 3 labs | contradictoires (acc.) |
|---|---|---|---|---|---|---|---|---|---|
| A. Profil seul | 0.622 (11 %) | 0.656 (10 %) | 0.531 (31 %) | 0.442 (37 %) | 0.660 (0 %) | 0.660 (0 %) | 0.660 (0 %) | 0.660 (0 %) | 0.635 (n=331) |
| B. Labs seuls | 0.821 (0 %) | 0.821 (0 %) | 0.821 (0 %) | 0.821 (0 %) | 0.020 (90 %) | 0.242 (72 %) | 0.402 (56 %) | 0.510 (41 %) | 0.833 (n=331) |
| C. CareerNet | 0.869 (4 %) | 0.852 (4 %) | 0.811 (11 %) | 0.781 (11 %) | 0.620 (31 %) | 0.663 (27 %) | 0.700 (22 %) | 0.729 (17 %) | 0.860 (n=331) |
| D. CareerNet à porte | 0.866 (3 %) | 0.859 (2 %) | 0.825 (8 %) | 0.797 (10 %) | 0.644 (29 %) | 0.657 (28 %) | 0.688 (23 %) | 0.705 (19 %) | 0.860 (n=331) |
| C+. CareerNet + labels doux | 0.860 (4 %) | 0.853 (4 %) | 0.816 (11 %) | 0.784 (12 %) | 0.620 (31 %) | 0.658 (27 %) | 0.696 (23 %) | 0.722 (18 %) | 0.862 (n=331) |
| E. CareerNet sans plaisir | 0.803 (6 %) | 0.785 (5 %) | 0.736 (15 %) | 0.702 (16 %) | 0.625 (27 %) | 0.632 (26 %) | 0.674 (22 %) | 0.689 (18 %) | 0.788 (n=331) |

## Choix du modèle (validation uniquement)

| Config | score de sélection |
|---|---|
| C. CareerNet | 0.7705 |
| D. CareerNet à porte | 0.7637 |
| B. Labs seuls | 0.6723 |
| A. Profil seul | 0.6076 |

**Retenu : C. CareerNet**

## Porte de fusion (D, graine 0) : valeur moyenne g selon le nombre de labs faits (test)

g proche de 1 = la représentation du profil domine ; proche de 0 = celle des labs.

| labs faits | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| g moyen | 0.80 | 0.74 | 0.72 | 0.72 | 0.70 | 0.68 | 0.69 | 0.72 | 0.58 |
