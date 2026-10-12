# Fusion profil / labs — dataset v1 (test n = 500, graines [0, 1, 2])

## Performances sur données d'origine (test)

| Config | Accuracy | Balanced acc. | Macro-F1 | Top-3 | NLL | ECE | ECE après T |
|---|---|---|---|---|---|---|---|
| A. Profil seul | 0.797 ± 0.007 | 0.789 ± 0.006 | 0.776 ± 0.008 | 0.975 ± 0.005 | 0.580 ± 0.011 | 0.058 ± 0.027 | 0.049 ± 0.023 |
| B. Labs seuls | 0.816 ± 0.002 | 0.808 ± 0.003 | 0.808 ± 0.004 | 0.952 ± 0.012 | 0.567 ± 0.009 | 0.045 ± 0.009 | 0.046 ± 0.007 |
| C. CareerNet | 0.909 ± 0.003 | 0.909 ± 0.003 | 0.903 ± 0.007 | 0.989 ± 0.002 | 0.297 ± 0.012 | 0.045 ± 0.001 | 0.026 ± 0.002 |
| D. CareerNet à porte | 0.902 ± 0.002 | 0.899 ± 0.006 | 0.895 ± 0.005 | 0.991 ± 0.004 | 0.310 ± 0.008 | 0.044 ± 0.002 | 0.033 ± 0.007 |
| C+. CareerNet + labels doux | 0.905 ± 0.013 | 0.901 ± 0.011 | 0.896 ± 0.014 | 0.995 ± 0.001 | 0.358 ± 0.014 | 0.111 ± 0.007 | 0.025 ± 0.002 |
| E. CareerNet sans plaisir | 0.873 ± 0.005 | 0.866 ± 0.017 | 0.859 ± 0.010 | 0.982 ± 0.000 | 0.407 ± 0.008 | 0.034 ± 0.001 | 0.035 ± 0.009 |

## Robustesse (test, macro-F1 ; entre parenthèses : % d'étudiants dont le métier n°1 change)

| Config | self +2 | self −2 | bruit σ=2 | inflation autre métier | 0 lab | 1 lab | 2 labs | 3 labs | contradictoires (acc.) |
|---|---|---|---|---|---|---|---|---|---|
| A. Profil seul | 0.753 (8 %) | 0.769 (7 %) | 0.627 (27 %) | 0.549 (34 %) | 0.776 (0 %) | 0.776 (0 %) | 0.776 (0 %) | 0.776 (0 %) | 0.783 (n=315) |
| B. Labs seuls | 0.808 (0 %) | 0.808 (0 %) | 0.808 (0 %) | 0.808 (0 %) | 0.020 (91 %) | 0.246 (74 %) | 0.429 (51 %) | 0.551 (38 %) | 0.820 (n=315) |
| C. CareerNet | 0.894 (4 %) | 0.900 (3 %) | 0.810 (14 %) | 0.747 (20 %) | 0.753 (18 %) | 0.788 (16 %) | 0.804 (13 %) | 0.812 (11 %) | 0.923 (n=315) |
| D. CareerNet à porte | 0.891 (3 %) | 0.899 (3 %) | 0.810 (13 %) | 0.738 (20 %) | 0.759 (17 %) | 0.777 (16 %) | 0.805 (13 %) | 0.807 (10 %) | 0.905 (n=315) |
| C+. CareerNet + labels doux | 0.904 (4 %) | 0.897 (3 %) | 0.801 (15 %) | 0.739 (20 %) | 0.739 (19 %) | 0.788 (16 %) | 0.813 (13 %) | 0.818 (10 %) | 0.921 (n=315) |
| E. CareerNet sans plaisir | 0.850 (4 %) | 0.846 (4 %) | 0.740 (18 %) | 0.673 (24 %) | 0.764 (13 %) | 0.772 (13 %) | 0.812 (9 %) | 0.791 (9 %) | 0.876 (n=315) |

## Choix du modèle (validation uniquement)

| Config | score de sélection |
|---|---|
| C. CareerNet | 0.8278 |
| D. CareerNet à porte | 0.8257 |
| A. Profil seul | 0.7255 |
| B. Labs seuls | 0.6871 |

**Retenu : C. CareerNet**

## Porte de fusion (D, graine 0) : valeur moyenne g selon le nombre de labs faits (test)

g proche de 1 = la représentation du profil domine ; proche de 0 = celle des labs.

| labs faits | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| g moyen | 0.85 | 0.80 | 0.79 | 0.78 | 0.77 | 0.76 | 0.77 | 0.79 | 0.69 |
