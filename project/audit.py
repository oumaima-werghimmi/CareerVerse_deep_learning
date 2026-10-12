"""
CareerVerse — Audit du dataset (lecture seule)
==============================================

  python audit.py --data data/careerverse_students.csv --tag v2
  python audit.py --data _v1_backup/data/careerverse_students.csv --tag v1

Mesure, sans rien modifier :
  1. qualité : doublons, manquants, bornes, déséquilibre, incohérences completed/score ;
  2. ce que mesurent les auto-évaluations : corrélation self_* <-> aptitude latente
     (data/careerverse_students_latent.csv, identique en v1 et v2) ;
  3. correspondance self_* <-> labs, via les poids LABS du générateur (seule correspondance
     justifiée : c'est celle qui a produit les scores) ;
  4. fuites possibles : pouvoir prédictif de chaque groupe de variables pris SEUL
     (régression logistique, validation croisée 5 plis) ;
  5. labels doux fit_* : accord avec career_label et formule de calcul.
Sortie : results/audit_<tag>.json
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from step2_generate_students import FEATS, LABS

LATENT = "data/careerverse_students_latent.csv"
METRICS = ["score", "time_min", "attempts", "hints_used", "enjoyment_1_5"]


def group_power(X, y, seed=0):
    """Macro-F1 en validation croisée d'une régression logistique sur un groupe de variables."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return round(float(f1_score(y, pred, average="macro")), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/careerverse_students.csv")
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    df = pd.read_csv(a.data)
    lat = pd.read_csv(LATENT)
    assert (lat.student_id == df.student_id).all() and (lat.career_label == df.career_label).all()
    y = df.career_label.values
    out = {"tag": a.tag, "n": len(df)}

    # ---------- 1. qualité
    labs = list(LABS)
    incoh = {}
    for l in labs:
        s, c = df[f"{l}__score"], df[f"{l}__completed"]
        bad = s.notna() & ((s >= 50).astype(float) != c)
        incoh[l] = {"n": int(bad.sum()), "scores": sorted(s[bad].round(2).unique().tolist())}
    out["quality"] = {
        "duplicate_ids": int(df.student_id.duplicated().sum()),
        "duplicate_rows": int(df.drop(columns="student_id").duplicated().sum()),
        "missing_outside_labs": int(df[[c for c in df if "__" not in c]].isna().sum().sum()),
        "labs_not_done_pct": round(float(df.filter(like="__score").isna().mean().mean() * 100), 1),
        "students_without_lab": int((df.n_labs_done == 0).sum()),
        "n_labs_done_matches_mask": bool((df.filter(like="__score").notna().sum(1) == df.n_labs_done).all()),
        "class_counts": df.career_label.value_counts().to_dict(),
        "completed_vs_score_mismatch": {k: v for k, v in incoh.items() if v["n"]},
    }

    # ---------- 2. self_* vs aptitude latente
    out["self_vs_latent_corr"] = {
        f: round(float(np.corrcoef(df[f"self_{f}"], lat[f"latent_{f}"])[0, 1]), 3) for f in FEATS}
    out["self_minus_latent_mean"] = round(float((df[[f"self_{f}" for f in FEATS]].values
                                                 - lat[[f"latent_{f}" for f in FEATS]].values).mean()), 3)

    # ---------- 3. self_* vs labs (correspondance = poids du générateur)
    lab_rel = {}
    for l, w in LABS.items():
        done = df[f"{l}__score"].notna()
        self_mix = sum(wt * df.loc[done, f"self_{f}"] for f, wt in w.items())
        lat_mix = sum(wt * lat.loc[done, f"latent_{f}"] for f, wt in w.items())
        sc = df.loc[done, f"{l}__score"]
        lab_rel[l] = {"corr_self_mix_score": round(float(np.corrcoef(self_mix, sc)[0, 1]), 3),
                      "corr_latent_mix_score": round(float(np.corrcoef(lat_mix, sc)[0, 1]), 3),
                      "corr_self_mix_latent_mix": round(float(np.corrcoef(self_mix, lat_mix)[0, 1]), 3)}
    out["self_vs_labs"] = lab_rel

    # ---------- 4. pouvoir prédictif de chaque groupe pris seul
    mask = df[[f"{l}__score" for l in labs]].notna().astype(float).values
    def lab_block(metric):
        return np.nan_to_num(df[[f"{l}__{metric}" for l in labs]].values, nan=0.0)
    groups = {
        "self_* (13)": df[[f"self_{f}" for f in FEATS]].values,
        "interest_* (5)": df.filter(like="interest_").values,
        "notes + année (5)": df[["year_of_study", "gpa_20", "grade_math_20", "grade_programming_20", "grade_networks_20"]].values,
        "masque labs faits (9)": mask,
        "lab scores + masque": np.hstack([lab_block("score"), mask]),
        "lab enjoyment + masque": np.hstack([lab_block("enjoyment_1_5"), mask]),
        "lab score/temps/essais/indices + masque": np.hstack([lab_block(m) for m in METRICS[:4]] + [mask]),
        "aptitudes latentes (référence, cachées)": lat[[f"latent_{f}" for f in FEATS]].values,
    }
    out["group_macro_f1_cv5"] = {g: group_power(X, y) for g, X in groups.items()}
    out["chance_macro_f1"] = round(1 / df.career_label.nunique(), 3)

    # ---------- 5. labels doux
    fit = df[[c for c in df if c.startswith("fit_")]]
    out["soft_labels"] = {
        "argmax_equals_label": round(float((fit.idxmax(axis=1).str[4:] == df.career_label).mean()), 3),
        "computed_from": "latent aptitudes vs career profiles (step2: exp(-mean sq. distance / 2)), not from observed inputs",
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/audit_{a.tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
