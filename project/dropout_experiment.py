"""
CareerVerse — Effet du « lab dropout » sur les profils avec peu de labs
=======================================================================

  python dropout_experiment.py --art artifacts --tag v2

Constat de fusion_experiments.py : sans aucun lab, CareerNet (lab dropout 0.15) fait moins bien
que le modèle « profil seul ». Hypothèse : à l'entraînement, le modèle voit trop rarement des
étudiants sans lab. On compare lab_dropout ∈ {0.15, 0.3, 0.5} avec la même règle de sélection
que fusion_experiments.py (validation uniquement), 3 graines.
Sortie : results/fusion_<tag>/dropout.json et dropout.md
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from fusion_experiments import Perturber, calibration, evaluate, fit_temperature
from train import load_split, model_kwargs, predict_proba, save_checkpoint, train

RATES = [0.15, 0.3, 0.5]
SEEDS = [0, 1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts")
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    out_dir = os.path.join("results", f"fusion_{a.tag}")
    meta = json.load(open(os.path.join(a.art, "preprocess.json"), encoding="utf-8"))
    npz = np.load(os.path.join(a.art, "dataset.npz"))
    va, te = load_split(npz, "val"), load_split(npz, "test")
    C = len(meta["classes"])
    pert = Perturber(meta, pd.read_csv("data/career_profiles.csv", index_col="career"))

    rows = []
    for rate in RATES:
        for seed in SEEDS:
            model, _, info = train("careernet", seed, lab_dropout=rate, art=a.art, verbose=False)
            T = fit_temperature(model, va)
            ev_val, ev_te = evaluate(model, va, pert, C), evaluate(model, te, pert, C)
            cal_T = calibration(te["y"].numpy(), predict_proba(model, te, temperature=T))
            save_checkpoint(os.path.join(out_dir, "models", f"careernet_drop{rate}_s{seed}.pt"), model,
                            "careernet", model_kwargs("careernet", meta), meta, info, temperature=T)
            row = {"lab_dropout": rate, "seed": seed, "temperature": T,
                   "selection_val": (ev_val["clean"]["macro_f1"] + ev_val["self_inflate_other"]["macro_f1"]
                                     + ev_val["labs_2"]["macro_f1"]) / 3}
            for k in ["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy", "nll", "ece"]:
                row[f"test_{k}"] = ev_te["clean"][k]
            row["test_ece_T"] = cal_T["ece"]
            for p in ["self_inflate_other", "self_noise2", "labs_0", "labs_1", "labs_2", "labs_3"]:
                row[f"test_{p}_f1"] = ev_te[p]["macro_f1"]
            rows.append(row)
            print(f"dropout {rate} seed {seed}: sel {row['selection_val']:.3f} | test F1 {row['test_macro_f1']:.3f} "
                  f"| 0 lab {row['test_labs_0_f1']:.3f} | 2 labs {row['test_labs_2_f1']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    g = df.drop(columns="seed").groupby("lab_dropout")
    mean, std = g.mean(), g.std()
    with open(os.path.join(out_dir, "dropout.json"), "w", encoding="utf-8") as f:
        json.dump({"mean": mean.round(4).to_dict(orient="index"), "std": std.round(4).to_dict(orient="index")},
                  f, indent=2)
    cols = [("selection_val", "Sélection (val)"), ("test_macro_f1", "Macro-F1"), ("test_top3_accuracy", "Top-3"),
            ("test_ece_T", "ECE après T"), ("test_self_inflate_other_f1", "Inflation"),
            ("test_labs_0_f1", "0 lab"), ("test_labs_1_f1", "1 lab"), ("test_labs_2_f1", "2 labs"),
            ("test_labs_3_f1", "3 labs")]
    md = [f"# Lab dropout — dataset {a.tag} (CareerNet, 3 graines)\n",
          "| lab dropout | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for r in mean.index:
        md.append(f"| {r} | " + " | ".join(f"{mean.loc[r, c]:.3f} ± {std.loc[r, c]:.3f}" for c, _ in cols) + " |")
    with open(os.path.join(out_dir, "dropout.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
