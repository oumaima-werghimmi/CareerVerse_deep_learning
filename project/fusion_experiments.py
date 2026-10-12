"""
CareerVerse — Expériences de fusion profil / labs et tests de robustesse
========================================================================

  python fusion_experiments.py --art artifacts    --tag v2
  python fusion_experiments.py --art artifacts_v1 --tag v1

Configurations comparées (3 graines chacune) :
  A. profil seul        B. labs seuls        C. CareerNet (concaténation)
  D. CareerNet à porte  C+. CareerNet + labels doux (0.3)
  E. CareerNet sans la mesure « plaisir » (diagnostic : cette mesure est générée à partir du label)

Protocole de robustesse : on modifie SEULEMENT les entrées des étudiants de validation et de test ;
les labels de référence ne changent pas. Ce sont des simulations de sensibilité, pas de nouvelles
vérités terrain.
  - self +2 / self −2 / bruit N(0, 2) : toutes les auto-évaluations, bornées à [0, 10] ;
  - inflation ciblée : +4 sur les 3 compétences signature d'un AUTRE métier que le vrai
    (tiré au hasard ; le label sert uniquement à choisir un métier différent) ;
  - au plus k labs (k = 0, 1, 2, 3) ;
  - sous-ensemble « contradictoire » (données non modifiées) : étudiants dont au moins un lab
    s'écarte de 25 points ou plus du score attendu d'après leurs auto-évaluations, avec la
    correspondance compétences -> lab du générateur (seule correspondance justifiée).
On rapporte le macro-F1 et le taux de changement du métier n°1 par rapport aux entrées d'origine.

Calibration : NLL, Brier, ECE (15 intervalles), avant et après temperature scaling
(température ajustée sur la VALIDATION).

Règle de choix du modèle, fixée avant de regarder le test (validation uniquement) :
  score = moyenne sur les graines de [macro-F1 val + macro-F1 val inflation ciblée
          + macro-F1 val avec au plus 2 labs] / 3,
  parmi A, B, C, D (C+ exclu : les labels doux n'existent pas sur de vraies données ;
  E exclu : diagnostic). En cas d'écart < 0,005, on garde le modèle le plus simple (C avant D).
Sorties : results/fusion_<tag>/ (runs.csv, summary.json, summary.md, models/*.pt)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from step2_generate_students import LABS as LAB_WEIGHTS
from train import load_split, metrics, predict_proba, save_checkpoint, train, model_kwargs

SEEDS = [0, 1, 2]
CONFIGS = [
    ("A. Profil seul", "profile", 0.0, ()),
    ("B. Labs seuls", "labs", 0.0, ()),
    ("C. CareerNet", "careernet", 0.0, ()),
    ("D. CareerNet à porte", "gated", 0.0, ()),
    ("C+. CareerNet + labels doux", "careernet", 0.3, ()),
    ("E. CareerNet sans plaisir", "careernet", 0.0, ("enjoyment_1_5",)),
]
SELECTABLE = ["A. Profil seul", "B. Labs seuls", "C. CareerNet", "D. CareerNet à porte"]
SIMPLICITY = {"A. Profil seul": 0, "B. Labs seuls": 0, "C. CareerNet": 1, "D. CareerNet à porte": 2}
PERTURBATIONS = ["clean", "self_up2", "self_down2", "self_noise2", "self_inflate_other",
                 "labs_0", "labs_1", "labs_2", "labs_3"]


# ---------------------------------------------------------------- calibration
def calibration(y, proba, bins=15):
    conf, pred = proba.max(1), proba.argmax(1)
    ece, edges = 0.0, np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            ece += sel.mean() * abs((pred[sel] == y[sel]).mean() - conf[sel].mean())
    onehot = np.eye(proba.shape[1])[y]
    return {"nll": float(-np.log(proba[np.arange(len(y)), y] + 1e-12).mean()),
            "brier": float(((proba - onehot) ** 2).sum(1).mean()),
            "ece": float(ece)}


@torch.no_grad()
def fit_temperature(model, d):
    logits = torch.cat([model(d["static"], d["lab"], d["mask"])])
    best_t, best = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.3), np.log(5), 120)):
        nll = torch.nn.functional.cross_entropy(logits / t, d["y"]).item()
        if nll < best:
            best_t, best = float(t), nll
    return best_t


# ---------------------------------------------------------------- perturbations
class Perturber:
    def __init__(self, meta, profiles):
        self.meta = meta
        sc = meta["static_cols"]
        self.self_idx = [i for i, c in enumerate(sc) if c.startswith("self_")]
        self.skills = [sc[i][5:] for i in self.self_idx]
        self.mean = np.array(meta["static_mean"])[self.self_idx]
        self.std = np.array(meta["static_std"])[self.self_idx]
        self.classes = meta["classes"]
        # 3 compétences signature de chaque métier (profil O*NET + expert)
        self.signature = {c: [self.skills.index(s) for s in profiles.loc[c, self.skills].astype(float).nlargest(3).index]
                          for c in self.classes}
        lab_names = meta["labs"]
        self.lab_w = {i: {self.skills.index(s): w for s, w in LAB_WEIGHTS[l].items()} for i, l in enumerate(lab_names)}
        j = meta["lab_metrics"].index("score")
        self.score_mean, self.score_std = meta["lab_mean"][j], meta["lab_std"][j]
        self.score_j = j

    def raw_self(self, d):
        return d["static"][:, self.self_idx].numpy() * self.std + self.mean

    def with_self(self, d, raw):
        s = d["static"].clone()
        s[:, self.self_idx] = torch.tensor((np.clip(np.round(raw), 0, 10) - self.mean) / self.std, dtype=torch.float32)
        return {**d, "static": s}

    def apply(self, d, name, seed=0):
        rng = np.random.default_rng(seed)
        if name == "clean":
            return d
        raw = self.raw_self(d)
        if name == "self_up2":
            return self.with_self(d, raw + 2)
        if name == "self_down2":
            return self.with_self(d, raw - 2)
        if name == "self_noise2":
            return self.with_self(d, raw + rng.normal(0, 2, raw.shape))
        if name == "self_inflate_other":
            y = d["y"].numpy()
            other = (y + rng.integers(1, len(self.classes), len(y))) % len(self.classes)
            for i, c in enumerate(other):
                raw[i, self.signature[self.classes[c]]] += 4
            return self.with_self(d, raw)
        if name.startswith("labs_"):
            k = int(name.split("_")[1])
            m = d["mask"].clone()
            for i in range(len(m)):
                done = torch.nonzero(m[i]).flatten().numpy()
                if len(done) > k:
                    m[i, rng.choice(done, len(done) - k, replace=False)] = False
            return {**d, "mask": m, "lab": d["lab"] * m[..., None]}
        raise ValueError(name)

    def contradictory(self, d, gap=25):
        """Masque des étudiants dont un lab fait contredit l'auto-évaluation (données d'origine)."""
        raw = self.raw_self(d)
        score = d["lab"][:, :, self.score_j].numpy() * self.score_std + self.score_mean
        mask = d["mask"].numpy()
        flag = np.zeros(len(raw), dtype=bool)
        for li, w in self.lab_w.items():
            level = sum(wt * raw[:, si] for si, wt in w.items())
            expected = 100 / (1 + np.exp(-(level - 5.2) * 0.9))
            flag |= mask[:, li] & (np.abs(score[:, li] - expected) >= gap)
        return flag


def subset(d, sel):
    idx = torch.tensor(np.nonzero(sel)[0])
    return {k: v[idx] for k, v in d.items()}


def evaluate(model, d, pert, C, temperature=1.0):
    out = {}
    base_pred = predict_proba(model, d).argmax(1)
    y = d["y"].numpy()
    for name in PERTURBATIONS:
        p = predict_proba(model, pert.apply(d, name, seed=123), temperature=temperature)
        out[name] = {"macro_f1": float(f1_score(y, p.argmax(1), average="macro")),
                     "flip_rate": float((p.argmax(1) != base_pred).mean())}
        if name == "clean":
            out[name].update({k: float(v) for k, v in metrics(y, p, C).items()})
            out[name].update(calibration(y, p))
    sel = pert.contradictory(d)
    if sel.sum():
        sd = subset(d, sel)
        p = predict_proba(model, sd)
        out["contradictory"] = {"n": int(sel.sum()), "accuracy": float((p.argmax(1) == sd["y"].numpy()).mean()),
                                "macro_f1": float(f1_score(sd["y"].numpy(), p.argmax(1), average="macro"))}
    return out


@torch.no_grad()
def gate_by_labs(model, d):
    g, _, _ = model.gate_values(d["static"], d["lab"], d["mask"])
    n = d["mask"].sum(1).numpy()
    g = g.mean(1).numpy()
    return {int(k): round(float(g[n == k].mean()), 3) for k in np.unique(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="artifacts")
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    a = ap.parse_args()
    out_dir = os.path.join("results", f"fusion_{a.tag}")
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)

    meta = json.load(open(os.path.join(a.art, "preprocess.json"), encoding="utf-8"))
    npz = np.load(os.path.join(a.art, "dataset.npz"))
    va, te = load_split(npz, "val"), load_split(npz, "test")
    C = len(meta["classes"])
    pert = Perturber(meta, pd.read_csv("data/career_profiles.csv", index_col="career"))

    rows, gates = [], {}
    for name, kind, sw, drop in CONFIGS:
        for seed in a.seeds:
            model, hist, info = train(kind, seed, soft_weight=sw, art=a.art, drop_metrics=drop, verbose=False)
            T = fit_temperature(model, va)
            ev_val, ev_te = evaluate(model, va, pert, C), evaluate(model, te, pert, C)
            p_te_T = predict_proba(model, te, temperature=T)
            cal_T = calibration(te["y"].numpy(), p_te_T)
            slug = f"{kind}_sw{sw}_{'-'.join(drop) or 'all'}_s{seed}"
            save_checkpoint(os.path.join(out_dir, "models", f"{slug}.pt"), model, kind,
                            model_kwargs(kind, meta, drop), meta, info, temperature=T)
            row = {"config": name, "seed": seed, "epochs": info["epochs_run"], "params": info["n_params"],
                   "temperature": T, "val_macro_f1": ev_val["clean"]["macro_f1"],
                   "val_inflate_f1": ev_val["self_inflate_other"]["macro_f1"],
                   "val_labs2_f1": ev_val["labs_2"]["macro_f1"]}
            for k in ["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy", "nll", "brier", "ece"]:
                row[f"test_{k}"] = ev_te["clean"][k]
            row.update({"test_ece_T": cal_T["ece"], "test_nll_T": cal_T["nll"]})
            for p in PERTURBATIONS[1:]:
                row[f"test_{p}_f1"] = ev_te[p]["macro_f1"]
                row[f"test_{p}_flip"] = ev_te[p]["flip_rate"]
            if "contradictory" in ev_te:
                row["test_contra_n"] = ev_te["contradictory"]["n"]
                row["test_contra_acc"] = ev_te["contradictory"]["accuracy"]
                row["test_contra_f1"] = ev_te["contradictory"]["macro_f1"]
            rows.append(row)
            if kind == "gated" and seed == a.seeds[0]:
                gates = gate_by_labs(model, te)
            print(f"{name:30s} seed {seed}: val F1 {row['val_macro_f1']:.3f} | test F1 {row['test_macro_f1']:.3f} "
                  f"| inflate {row['test_self_inflate_other_f1']:.3f} | 0 lab {row['test_labs_0_f1']:.3f} "
                  f"| ECE {row['test_ece']:.3f}->{row['test_ece_T']:.3f}", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out_dir, "runs.csv"), index=False)
    num = runs.drop(columns=["seed"]).groupby("config", sort=False)
    mean, std = num.mean(numeric_only=True), num.std(numeric_only=True).fillna(0)

    sel_score = (mean["val_macro_f1"] + mean["val_inflate_f1"] + mean["val_labs2_f1"]) / 3
    cands = sel_score[SELECTABLE].sort_values(ascending=False)
    top = cands.index[0]
    close = [c for c in cands.index if cands[top] - cands[c] < 0.005]
    chosen = min(close, key=lambda c: (SIMPLICITY[c], -cands[c]))

    # rapport détaillé du modèle retenu (graine 0) : par métier + matrice de confusion
    kind = dict((c[0], c[1]) for c in CONFIGS)[chosen]
    ck = torch.load(os.path.join(out_dir, "models", f"{kind}_sw0.0_all_s{a.seeds[0]}.pt"), map_location="cpu")
    from train import make_model
    best = make_model(ck["kind"], ck["model_kwargs"])
    best.load_state_dict(ck["state_dict"])
    p = predict_proba(best, te)
    rep = classification_report(te["y"].numpy(), p.argmax(1), target_names=meta["classes"], output_dict=True)
    cm = confusion_matrix(te["y"].numpy(), p.argmax(1)).tolist()

    summary = {"tag": a.tag, "seeds": a.seeds, "selection_score_val": cands.round(4).to_dict(), "chosen": chosen,
               "mean": mean.round(4).to_dict(orient="index"), "std": std.round(4).to_dict(orient="index"),
               "gate_mean_by_n_labs_test": gates,
               "chosen_per_class_test": {c: {k: round(rep[c][k], 3) for k in ["precision", "recall", "f1-score", "support"]}
                                         for c in meta["classes"]},
               "chosen_confusion_test": cm}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    def ms(c, k, d=3):
        return f"{mean.loc[c, k]:.{d}f} ± {std.loc[c, k]:.{d}f}"
    md = [f"# Fusion profil / labs — dataset {a.tag} (test n = {len(te['y'])}, graines {a.seeds})\n",
          "## Performances sur données d'origine (test)\n",
          "| Config | Accuracy | Balanced acc. | Macro-F1 | Top-3 | NLL | ECE | ECE après T |", "|---|---|---|---|---|---|---|---|"]
    for c in mean.index:
        md.append(f"| {c} | {ms(c,'test_accuracy')} | {ms(c,'test_balanced_accuracy')} | {ms(c,'test_macro_f1')} | "
                  f"{ms(c,'test_top3_accuracy')} | {ms(c,'test_nll')} | {ms(c,'test_ece')} | {ms(c,'test_ece_T')} |")
    md += ["\n## Robustesse (test, macro-F1 ; entre parenthèses : % d'étudiants dont le métier n°1 change)\n",
           "| Config | self +2 | self −2 | bruit σ=2 | inflation autre métier | 0 lab | 1 lab | 2 labs | 3 labs | contradictoires (acc.) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in mean.index:
        cells = [f"{mean.loc[c, f'test_{p}_f1']:.3f} ({100*mean.loc[c, f'test_{p}_flip']:.0f} %)" for p in PERTURBATIONS[1:]]
        contra = f"{mean.loc[c,'test_contra_acc']:.3f} (n={int(mean.loc[c,'test_contra_n'])})" if "test_contra_acc" in mean else "-"
        md.append(f"| {c} | " + " | ".join(cells) + f" | {contra} |")
    md += ["\n## Choix du modèle (validation uniquement)\n", "| Config | score de sélection |", "|---|---|"]
    md += [f"| {c} | {v:.4f} |" for c, v in cands.items()]
    md += [f"\n**Retenu : {chosen}**"]
    if gates:
        md += ["\n## Porte de fusion (D, graine 0) : valeur moyenne g selon le nombre de labs faits (test)\n",
               "g proche de 1 = la représentation du profil domine ; proche de 0 = celle des labs.\n",
               "| labs faits | " + " | ".join(str(k) for k in gates) + " |", "|---" * (len(gates) + 1) + "|",
               "| g moyen | " + " | ".join(f"{v:.2f}" for v in gates.values()) + " |"]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
