"""
CareerVerse — Étape 4 : entraînement
====================================

  python train.py                                  # CareerNet, réglages par défaut
  python train.py --model mlp                      # baseline MLP
  python train.py --model gated                    # CareerNet avec fusion par porte
  python train.py --soft_weight 0.3                # avec labels doux fit_* (données simulées uniquement)
  python train.py --art artifacts_v1               # autre jeu de données prétraité (preprocess.py --out)

Techniques utilisées (à citer dans le rapport) :
  - perte = CrossEntropy pondérée par classe + label smoothing (0.05)
            + soft_weight * KL(fit_* || prédiction)   (apprentissage multi-cible)
    F.kl_div(log_softmax(logits), soft) calcule KL(soft || prédiction) = Σ soft·(log soft − log p) :
    c'est la direction « le modèle doit couvrir la distribution douce ».
  - "lab dropout" : pendant l'entraînement, on masque au hasard 15 % des labs faits
    -> le modèle apprend à prédire avec des profils incomplets (augmentation de données)
  - AdamW + weight decay, scheduler cosinus avec warm-up, gradient clipping
  - early stopping sur le macro-F1 de validation (patience 25 époques)

Le checkpoint enregistre aussi le prétraitement (preprocess.json) et les paramètres du modèle :
predict.py n'utilise que ces informations, ce qui évite d'appliquer une normalisation qui ne
correspond pas aux poids (cas d'un preprocess.py relancé après l'entraînement).
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score, balanced_accuracy_score

from model import CareerNet, GatedCareerNet, LabsOnly, MLPBaseline, ProfileOnly

ART = "artifacts"
KINDS = ["careernet", "gated", "profile", "labs", "mlp"]


def load_split(npz, name, device="cpu"):
    t = lambda k, dt: torch.tensor(npz[f"{name}_{k}"], dtype=dt, device=device)
    return {"static": t("static", torch.float32), "lab": t("lab", torch.float32), "mask": t("mask", torch.bool),
            "y": t("y", torch.long), "soft": t("soft", torch.float32)}


def model_kwargs(kind, meta, drop_metrics=()):
    """Paramètres de construction, enregistrés dans le checkpoint pour pouvoir le recharger."""
    kw = {"n_static": len(meta["static_cols"]), "n_labs": len(meta["labs"]),
          "n_metrics": len(meta["lab_metrics"]), "n_classes": len(meta["classes"])}
    if drop_metrics:
        kw["use_metrics"] = [0.0 if m in drop_metrics else 1.0 for m in meta["lab_metrics"]]
    if kind == "profile":
        kw["static_keep"] = [0.0 if c == "n_labs_done" else 1.0 for c in meta["static_cols"]]
    return kw


def make_model(kind, kw):
    if kind == "mlp":
        return MLPBaseline(kw["n_static"] + kw["n_labs"] * kw["n_metrics"] + kw["n_labs"], kw["n_classes"])
    cls = {"careernet": CareerNet, "gated": GatedCareerNet, "profile": ProfileOnly, "labs": LabsOnly}[kind]
    return cls(**kw)


def meta_fingerprint(meta):
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:16]


@torch.no_grad()
def predict_proba(model, d, bs=1024, temperature=1.0):
    model.eval()
    out = [F.softmax(model(d["static"][i:i + bs], d["lab"][i:i + bs], d["mask"][i:i + bs]) / temperature, -1)
           for i in range(0, len(d["y"]), bs)]
    return torch.cat(out).cpu().numpy()


def metrics(y, proba, n_classes):
    pred = proba.argmax(1)
    return {"accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "top3_accuracy": top_k_accuracy_score(y, proba, k=3, labels=range(n_classes))}


def train(kind="careernet", seed=0, epochs=200, bs=128, lr=1e-3, wd=1e-4, soft_weight=0.0,
          lab_dropout=0.15, patience=25, out=None, verbose=True, art=ART, drop_metrics=()):
    torch.manual_seed(seed)
    np.random.seed(seed)
    meta = json.load(open(os.path.join(art, "preprocess.json"), encoding="utf-8"))
    npz = np.load(os.path.join(art, "dataset.npz"))
    tr, va = load_split(npz, "train"), load_split(npz, "val")
    C = len(meta["classes"])

    kw = model_kwargs(kind, meta, drop_metrics)
    model = make_model(kind, kw)
    class_w = torch.tensor(meta["class_weights"], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    steps_per_epoch = math.ceil(len(tr["y"]) / bs)
    total, warm = epochs * steps_per_epoch, 5 * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    best, best_state, wait, hist = -1, None, 0, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr["y"]))
        tot_loss = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            s, l, m, y, soft = tr["static"][b], tr["lab"][b], tr["mask"][b], tr["y"][b], tr["soft"][b]
            if lab_dropout > 0:
                m = m & (torch.rand_like(m, dtype=torch.float32) > lab_dropout)
                l = l * m[..., None]
            logits = model(s, l, m)
            loss = F.cross_entropy(logits, y, weight=class_w, label_smoothing=0.05)
            if soft_weight > 0:
                # KL(soft || prédiction), voir l'en-tête du fichier
                loss = loss + soft_weight * F.kl_div(F.log_softmax(logits, -1), soft, reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(b)

        pv = predict_proba(model, va)
        val_loss = F.nll_loss(torch.log(torch.tensor(pv) + 1e-9), va["y"]).item()
        mv = metrics(va["y"].numpy(), pv, C)
        hist.append({"epoch": ep, "train_loss": tot_loss / len(perm), "val_loss": val_loss, **mv})
        if mv["macro_f1"] > best:
            best, wait = mv["macro_f1"], 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"ep {ep:3d} | loss {hist[-1]['train_loss']:.3f} | val loss {val_loss:.3f} "
                  f"| val acc {mv['accuracy']:.3f} | val F1 {mv['macro_f1']:.3f}")
        if wait >= patience:
            if verbose:
                print(f"early stopping à l'époque {ep} (meilleur val F1 = {best:.3f})")
            break

    model.load_state_dict(best_state)
    info = {"kind": kind, "seed": seed, "epochs_run": ep, "best_val_macro_f1": best,
            "train_seconds": round(time.time() - t0, 1),
            "n_params": sum(p.numel() for p in model.parameters()),
            "hparams": dict(bs=bs, lr=lr, wd=wd, soft_weight=soft_weight, lab_dropout=lab_dropout,
                            drop_metrics=list(drop_metrics)),
            "art": art, "meta_fingerprint": meta_fingerprint(meta)}
    if out:
        save_checkpoint(out, model, kind, kw, meta, info)
    return model, hist, info


def save_checkpoint(out, model, kind, kw, meta, info, temperature=1.0):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "kind": kind, "model_kwargs": kw, "meta": meta,
                "temperature": temperature, "info": info}, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="careernet", choices=KINDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--soft_weight", type=float, default=0.0)
    ap.add_argument("--lab_dropout", type=float, default=0.15)
    ap.add_argument("--art", default=ART, help="dossier contenant dataset.npz et preprocess.json")
    ap.add_argument("--out", default=None, help="checkpoint (défaut : <art>/<model>.pt)")
    a = ap.parse_args()
    out = a.out or os.path.join(a.art, f"{a.model}.pt")
    model, hist, info = train(a.model, a.seed, a.epochs, soft_weight=a.soft_weight,
                              lab_dropout=a.lab_dropout, out=out, art=a.art)
    npz = np.load(os.path.join(a.art, "dataset.npz"))
    te = load_split(npz, "test")
    C = len(json.load(open(os.path.join(a.art, "preprocess.json"), encoding="utf-8"))["classes"])
    print("\nTEST :", {k: round(v, 4) for k, v in metrics(te["y"].numpy(), predict_proba(model, te), C).items()})
    print(info)
    print(f"checkpoint -> {out}")
