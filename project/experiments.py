"""
CareerVerse — Étape 5 : expériences, évaluation et figures
==========================================================

  python experiments.py            # ~10 min sur CPU

Compare sur le MÊME split de test :
  - Régression logistique, Gradient Boosting (baselines machine learning classiques)
  - MLP (baseline deep learning)
  - CareerNet avec labels doux  (soft_weight = 0.3)
  - CareerNet sans labels doux  (soft_weight = 0)  -> configuration réaliste pour de vraies données
Chaque modèle deep learning est entraîné avec 3 graines -> moyenne ± écart-type.

Analyses complémentaires sur le meilleur modèle :
  - robustesse : performance quand l'étudiant n'a fait que k labs
  - importance des groupes de variables (permutation)
  - équité : performance par genre (le genre n'est PAS une entrée du modèle)
  - rapport de classification par métier + matrice de confusion
Sorties : results/*.png, results/metrics.json, results/results.md
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from train import ART, load_split, metrics, predict_proba, train

RES = "results"
SEEDS = [0, 1, 2]
BEST = "CareerNet (sans labels doux)"  # modèle retenu : utilisable sur de vraies données (pas de fit_*)
# palette (référence dataviz) — encre, surfaces, séries
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlecolor": INK})


def flat(d):
    """Features aplaties pour les modèles sklearn (labs + masque)."""
    return np.hstack([d["static"].numpy(), d["lab"].flatten(1).numpy(), d["mask"].float().numpy()])


def sk_proba(m, d):
    return m.predict_proba(flat(d))


def mask_to_k(d, k, rng):
    """Garde au plus k labs faits par étudiant (tirés au hasard) — simule un profil incomplet."""
    m = d["mask"].clone()
    for i in range(len(m)):
        done = torch.nonzero(m[i]).flatten().numpy()
        if len(done) > k:
            drop = rng.choice(done, len(done) - k, replace=False)
            m[i, drop] = False
    return {**d, "mask": m, "lab": d["lab"] * m[..., None]}


def main():
    os.makedirs(RES, exist_ok=True)
    meta = json.load(open(os.path.join(ART, "preprocess.json")))
    classes = meta["classes"]
    C = len(classes)
    npz = np.load(os.path.join(ART, "dataset.npz"))
    tr, va, te = (load_split(npz, s) for s in ["train", "val", "test"])
    y_te = te["y"].numpy()
    rows, hist_best, models = [], None, {}

    # ---------------- baselines sklearn
    print("[1/4] baselines sklearn")
    Xtr = np.vstack([flat(tr), flat(va)])
    ytr = np.concatenate([tr["y"].numpy(), va["y"].numpy()])
    sk = {"Régression logistique": LogisticRegression(max_iter=5000, class_weight="balanced"),
          "Gradient Boosting": HistGradientBoostingClassifier(random_state=0, class_weight="balanced")}
    for name, m in sk.items():
        m.fit(Xtr, ytr)
        models[name] = m
        rows.append({"model": name, "seed": 0, **metrics(y_te, sk_proba(m, te), C)})

    # ---------------- modèles deep learning
    configs = [("MLP", "mlp", 0.0), ("CareerNet (sans labels doux)", "careernet", 0.0),
               ("CareerNet (+ labels doux)", "careernet", 0.3)]
    for i, (name, kind, sw) in enumerate(configs, start=2):
        print(f"[{i}/4] {name}")
        for seed in SEEDS:
            out = f"{ART}/careernet.pt" if (name == BEST and seed == 0) else None
            model, hist, info = train(kind, seed, soft_weight=sw, out=out, verbose=False)
            r = {"model": name, "seed": seed, **metrics(y_te, predict_proba(model, te), C),
                 "epochs": info["epochs_run"], "params": info["n_params"], "sec": info["train_seconds"]}
            rows.append(r)
            print(f"   seed {seed}: acc={r['accuracy']:.3f} F1={r['macro_f1']:.3f} ({info['epochs_run']} ép.)")
            if seed == 0:
                models[name] = model
                if out:
                    hist_best = hist

    res = pd.DataFrame(rows)
    res.to_csv(f"{RES}/all_runs.csv", index=False)
    order = ["Régression logistique", "Gradient Boosting", "MLP",
             "CareerNet (sans labels doux)", "CareerNet (+ labels doux)"]
    agg = res.groupby("model")[["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy"]].agg(["mean", "std"]).reindex(order)

    best_name = BEST
    best = models[best_name]
    proba = predict_proba(best, te)
    pred = proba.argmax(1)

    # ---------------- robustesse au nombre de labs
    print("[analyses] robustesse, importance, équité")
    rng = np.random.default_rng(0)
    ks = [0, 1, 2, 3, 4, 5, 9]
    robust = {n: [] for n in ["Gradient Boosting", "MLP", best_name]}
    for k in ks:
        dk = mask_to_k(te, k, np.random.default_rng(k))
        for n in robust:
            p = sk_proba(models[n], dk) if n in sk else predict_proba(models[n], dk)
            robust[n].append(f1_score(y_te, p.argmax(1), average="macro"))

    # ---------------- importance par permutation (groupes de variables)
    sc = meta["static_cols"]
    groups = {"Notes académiques": [i for i, c in enumerate(sc) if c.endswith("_20") or c == "year_of_study"],
              "Auto-évaluation": [i for i, c in enumerate(sc) if c.startswith("self_")],
              "Intérêts déclarés": [i for i, c in enumerate(sc) if c.startswith("interest_")]}
    base_f1 = f1_score(y_te, pred, average="macro")
    imp = {}
    for g, idx in groups.items():
        drops = []
        for r in range(5):
            s = te["static"].clone()
            perm = torch.tensor(rng.permutation(len(s)))
            s[:, idx] = s[perm][:, idx]
            drops.append(base_f1 - f1_score(y_te, predict_proba(best, {**te, "static": s}).argmax(1), average="macro"))
        imp[g] = float(np.mean(drops))
    drops = []
    for r in range(5):
        perm = torch.tensor(rng.permutation(len(te["y"])))
        d = {**te, "lab": te["lab"][perm], "mask": te["mask"][perm]}
        drops.append(base_f1 - f1_score(y_te, predict_proba(best, d).argmax(1), average="macro"))
    imp["Career Labs (tous)"] = float(np.mean(drops))
    for j, lab in enumerate(meta["labs"]):
        drops = []
        for r in range(5):
            perm = torch.tensor(rng.permutation(len(te["y"])))
            l, m = te["lab"].clone(), te["mask"].clone()
            l[:, j], m[:, j] = l[perm, j], m[perm, j]
            drops.append(base_f1 - f1_score(y_te, predict_proba(best, {**te, "lab": l, "mask": m}).argmax(1), average="macro"))
        imp[lab.replace("lab_", "Lab ")] = float(np.mean(drops))

    # ---------------- équité
    g_te = npz["test_gender"]
    fair = {g: {"n": int((g_te == g).sum()),
                "accuracy": float((pred[g_te == g] == y_te[g_te == g]).mean()),
                "macro_f1": float(f1_score(y_te[g_te == g], pred[g_te == g], average="macro"))}
            for g in sorted(set(g_te))}
    # intervalle de confiance bootstrap (95 %) de l'écart d'accuracy F - M
    correct = (pred == y_te)
    gaps = []
    for _ in range(2000):
        b = rng.integers(0, len(y_te), len(y_te))
        cb, gb = correct[b], g_te[b]
        gaps.append(cb[gb == "F"].mean() - cb[gb == "M"].mean())
    fair["gap_F_minus_M"] = {"accuracy_gap": round(fair["F"]["accuracy"] - fair["M"]["accuracy"], 4),
                             "ci95": [round(float(np.percentile(gaps, 2.5)), 4), round(float(np.percentile(gaps, 97.5)), 4)]}

    report = classification_report(y_te, pred, target_names=classes, output_dict=True)
    cm = confusion_matrix(y_te, pred, normalize="true")

    # ================= FIGURES
    short = {c: c.replace(" Developer", " Dev").replace(" Engineer", " Eng.").replace("Frontend/Web", "Frontend")
             .replace("Database Administrator", "DB Admin").replace("Cybersecurity Analyst", "Cybersécurité")
             .replace("System/Network Admin", "Sys/Net Admin") for c in classes}

    # 1. comparaison des modèles
    fig, ax = plt.subplots(figsize=(8, 3.6))
    m_, s_ = agg[("macro_f1", "mean")], agg[("macro_f1", "std")].fillna(0)
    colors = [MUTED, MUTED, "#86b6ef", BLUE, "#1c5cab"]
    for i, (v, sd, col) in enumerate(zip(m_, s_, colors)):
        ax.errorbar(v, i, xerr=sd if sd > 0 else None, fmt="o", ms=9, color=col, mec=SURF, mew=2,
                    ecolor=INK2, elinewidth=1.2, capsize=4)
        ax.text(v + max(sd, 0) + 0.003, i, f"{v:.3f}" + (f" ± {sd:.3f}" if sd > 0 else ""),
                va="center", color=INK, fontsize=9)
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0.86, 0.92)
    ax.set_xlabel("Macro-F1 sur le jeu de test (moyenne ± écart-type, 3 graines)")
    ax.set_title("Comparaison des modèles (points : moyenne, barres : ± écart-type)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{RES}/1_comparaison_modeles.png", dpi=160)
    plt.close(fig)

    # 2. courbes d'apprentissage
    h = pd.DataFrame(hist_best)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(h.epoch, h.train_loss, color=BLUE, lw=2, label="train")
    axes[0].plot(h.epoch, h.val_loss, color=ORANGE, lw=2, label="validation")
    axes[0].set_title("Perte")
    axes[0].set_xlabel("époque")
    axes[0].legend(frameon=False)
    axes[1].plot(h.epoch, h.macro_f1, color=BLUE, lw=2, label="macro-F1")
    axes[1].plot(h.epoch, h.accuracy, color=AQUA, lw=2, label="accuracy")
    be = h.macro_f1.idxmax()
    axes[1].scatter([h.epoch[be]], [h.macro_f1[be]], s=40, color=BLUE, edgecolor=SURF, lw=2, zorder=3)
    axes[1].annotate(f"meilleur modèle\népoque {h.epoch[be]}", (h.epoch[be], h.macro_f1[be]),
                     xytext=(10, -30), textcoords="offset points", color=INK2, fontsize=8.5)
    axes[1].set_ylim(0.6, 1)
    axes[1].set_title("Validation")
    axes[1].set_xlabel("époque")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("CareerNet — courbes d'apprentissage (early stopping)", fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(f"{RES}/2_courbes_apprentissage.png", dpi=160)
    plt.close(fig)

    # 3. matrice de confusion
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(7.5, 6.3))
    ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
    for i in range(C):
        for j in range(C):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]*100:.0f}", ha="center", va="center", fontsize=8.5,
                        color="white" if cm[i, j] > 0.55 else INK)
    ax.set_xticks(range(C), [short[c] for c in classes], rotation=40, ha="right")
    ax.set_yticks(range(C), [short[c] for c in classes])
    ax.set_xlabel("métier prédit")
    ax.set_ylabel("métier réel")
    ax.grid(False)
    ax.set_title("Matrice de confusion — CareerNet (% par ligne)")
    fig.tight_layout()
    fig.savefig(f"{RES}/3_matrice_confusion.png", dpi=160)
    plt.close(fig)

    # 4. robustesse
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xs = range(len(ks))
    for (n, col) in zip(robust, [MUTED, "#86b6ef", BLUE]):
        ax.plot(xs, robust[n], color=col, lw=2, marker="o", ms=5, label=n)
        ax.text(len(ks) - 1 + 0.1, robust[n][-1], f"{robust[n][-1]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(list(xs), [str(k) if k < 9 else "tous" for k in ks])
    ax.set_xlabel("nombre maximum de Career Labs faits par l'étudiant")
    ax.set_ylabel("macro-F1 (test)")
    ax.set_title("Robustesse aux profils incomplets")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{RES}/4_robustesse_labs.png", dpi=160)
    plt.close(fig)

    # 5. importance
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh([k for k, _ in items], [v for _, v in items],
            color=[BLUE if not k.startswith("Lab ") else "#86b6ef" for k, _ in items], height=0.6)
    for i, (k, v) in enumerate(items):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("baisse du macro-F1 quand le groupe est permuté")
    ax.set_title("Importance des sources d'information (CareerNet)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(f"{RES}/5_importance_variables.png", dpi=160)
    plt.close(fig)

    # ================= SAUVEGARDE
    out = {"summary": {m: {k: {"mean": round(agg.loc[m, (k, "mean")], 4), "std": round(float(np.nan_to_num(agg.loc[m, (k, "std")])), 4)}
                           for k in ["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy"]} for m in order},
           "robustness_macro_f1": {"k_labs": ks, **{n: [round(v, 4) for v in r] for n, r in robust.items()}},
           "permutation_importance": {k: round(v, 4) for k, v in imp.items()},
           "fairness_by_gender": fair,
           "per_class": {c: {k: round(report[c][k], 3) for k in ["precision", "recall", "f1-score", "support"]} for c in classes}}
    json.dump(out, open(f"{RES}/metrics.json", "w"), indent=2, ensure_ascii=False)

    md = ["# Résultats CareerVerse — test (n = %d)\n" % len(y_te), "## Comparaison des modèles\n",
          "| Modèle | Accuracy | Balanced acc. | Macro-F1 | Top-3 |", "|---|---|---|---|---|"]
    for m in order:
        f = lambda k: f"{agg.loc[m, (k, 'mean')]:.3f}" + (f" ± {agg.loc[m, (k, 'std')]:.3f}" if not np.isnan(agg.loc[m, (k, 'std')]) else "")
        md.append(f"| {m} | {f('accuracy')} | {f('balanced_accuracy')} | {f('macro_f1')} | {f('top3_accuracy')} |")
    md += [f"\n## Par métier ({BEST}, graine 0)\n", "| Métier | Précision | Rappel | F1 | n |", "|---|---|---|---|---|"]
    for c in classes:
        r = report[c]
        md.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {int(r['support'])} |")
    md += ["\n## Robustesse (macro-F1 selon le nombre max de labs faits)\n",
           "| Modèle | " + " | ".join(str(k) if k < 9 else "tous" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for n, r in robust.items():
        md.append(f"| {n} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    md += ["\n## Importance par permutation (baisse du macro-F1)\n", "| Groupe | Baisse |", "|---|---|"]
    for k, v in sorted(imp.items(), key=lambda kv: -kv[1]):
        md.append(f"| {k} | {v:.3f} |")
    md += ["\n## Équité (genre non utilisé en entrée)\n", "| Genre | n | Accuracy | Macro-F1 |", "|---|---|---|---|"]
    for g, v in fair.items():
        if g.startswith("gap"):
            md.append(f"\nÉcart d'accuracy F - M : {v['accuracy_gap']:+.3f}, IC 95 % bootstrap {v['ci95']}"
                      + (" -> contient 0 : pas d'écart significatif" if v['ci95'][0] < 0 < v['ci95'][1] else " -> écart significatif, à investiguer"))
            continue
        md.append(f"| {g} | {v['n']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
    open(f"{RES}/results.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
