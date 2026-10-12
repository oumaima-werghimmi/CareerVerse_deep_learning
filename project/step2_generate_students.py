"""
CareerVerse — Étape 2 : génération du dataset étudiants + Career Labs
=====================================================================

Tant que la plateforme n'a pas de vrais utilisateurs, on SIMULE des étudiants
dont le comportement est ancré dans les profils métiers O*NET (étape 1).

Principe (et c'est ce qui rend le label défendable) :
  1. On tire pour chaque étudiant un métier "vrai" latent  -> c'est le LABEL.
  2. On tire ses aptitudes réelles autour du profil O*NET de ce métier
     (+ un niveau général + du bruit individuel -> les classes se chevauchent).
  3. On génère ce que la plateforme OBSERVE : auto-évaluation (biaisée ; 25 % des
     étudiants surestiment les compétences d'un métier "rêvé" qui n'est pas le leur),
     notes académiques, intérêts déclarés (bruités), et résultats de Career Labs
     (score, temps, tentatives, indices, satisfaction) calculés à partir des
     aptitudes réelles + bruit. Tous les labs ne sont pas faits (valeurs manquantes).
  4. Le modèle ne voit JAMAIS les aptitudes latentes, seulement les observations.

Le label n'est donc PAS "le lab où il a eu le meilleur score" : c'est une vérité
terrain définie par le processus de simulation, documentée, et que le modèle doit
retrouver à partir de signaux bruités.

Sortie :
  data/careerverse_students.csv         dataset d'entraînement (features + label)
  data/careerverse_students_latent.csv  aptitudes cachées (pour analyse, PAS pour entraîner)
  data/data_dictionary.csv              description de chaque colonne

Usage :
  python step2_generate_students.py --n 5000 --seed 42
"""
import argparse
import os

import numpy as np
import pandas as pd

OUT_DIR = "data"
PROFILE_PATH = os.path.join(OUT_DIR, "career_profiles.csv")

# Profils de secours (à dire d'expert) si l'étape 1 n'a pas encore été lancée.
# => À REMPLACER par career_profiles.csv issu d'O*NET pour le rendu final.
FALLBACK_PROFILES = {
    #                        prog pyth linx dock clou netw secu sql  web  math test prob syst
    "Backend Developer":      [9,   7,   6,   5,   4,   3,   3,   8,   4,   5,   5,   8,   6],
    "Frontend/Web Developer": [8,   3,   3,   2,   2,   2,   2,   4,   10,  3,   5,   6,   4],
    "DevOps Engineer":        [7,   6,   9,   10,  8,   6,   5,   4,   2,   3,   5,   8,   9],
    "Cloud Engineer":         [6,   5,   8,   7,   10,  8,   6,   4,   2,   3,   3,   8,   9],
    "Cybersecurity Analyst":  [5,   6,   8,   3,   5,   8,   10,  4,   2,   4,   5,   9,   8],
    "Data Scientist":         [7,   10,  4,   3,   4,   1,   2,   7,   2,   10,  3,   9,   6],
    "Database Administrator": [5,   4,   6,   3,   5,   3,   5,   10,  1,   5,   4,   7,   7],
    "System/Network Admin":   [3,   4,   9,   4,   5,   10,  6,   4,   2,   3,   3,   7,   8],
    "QA/Test Engineer":       [6,   5,   4,   4,   2,   2,   3,   5,   6,   3,   10,  7,   6],
}
FEATS = ["programming", "python", "linux", "docker_k8s", "cloud", "networking", "security",
         "databases_sql", "web_frontend", "math_stats", "testing_quality",
         "problem_solving", "systems_thinking"]

# Career Labs CareerVerse : quelles aptitudes chaque lab mobilise (poids)
LABS = {
    "lab_backend_api":     {"programming": .4, "python": .2, "databases_sql": .2, "problem_solving": .2},
    "lab_web_ui":          {"web_frontend": .6, "programming": .3, "testing_quality": .1},
    "lab_devops_pipeline": {"docker_k8s": .4, "linux": .3, "systems_thinking": .2, "programming": .1},
    "lab_cloud_deploy":    {"cloud": .5, "networking": .2, "linux": .15, "systems_thinking": .15},
    "lab_security_ctf":    {"security": .5, "linux": .2, "networking": .15, "problem_solving": .15},
    "lab_data_analysis":   {"python": .35, "math_stats": .45, "databases_sql": .2},
    "lab_sql_database":    {"databases_sql": .6, "systems_thinking": .2, "problem_solving": .2},
    "lab_network_config":  {"networking": .55, "linux": .25, "systems_thinking": .2},
    "lab_qa_testing":      {"testing_quality": .55, "programming": .25, "problem_solving": .2},
}
# Lien lab -> métier(s) correspondant(s) : sert uniquement à simuler la motivation
LAB_DOMAIN = {
    "lab_backend_api": ["Backend Developer"], "lab_web_ui": ["Frontend/Web Developer"],
    "lab_devops_pipeline": ["DevOps Engineer"], "lab_cloud_deploy": ["Cloud Engineer"],
    "lab_security_ctf": ["Cybersecurity Analyst"], "lab_data_analysis": ["Data Scientist"],
    "lab_sql_database": ["Database Administrator"], "lab_network_config": ["System/Network Admin"],
    "lab_qa_testing": ["QA/Test Engineer"],
}
DREAMER_SHARE = 0.25   # part des étudiants qui surestiment les compétences d'un métier rêvé
LAB_NOISE = 5.0        # écart-type du bruit sur le score d'un lab (sur 100)

INTEREST_DOMAINS = ["software_dev", "web_design", "infrastructure", "security", "data_ai"]
CAREER_INTEREST = {
    "Backend Developer": "software_dev", "Frontend/Web Developer": "web_design",
    "DevOps Engineer": "infrastructure", "Cloud Engineer": "infrastructure",
    "Cybersecurity Analyst": "security", "Data Scientist": "data_ai",
    "Database Administrator": "data_ai", "System/Network Admin": "infrastructure",
    "QA/Test Engineer": "software_dev",
}


def load_profiles():
    if os.path.exists(PROFILE_PATH):
        p = pd.read_csv(PROFILE_PATH, index_col="career")
        p = p[FEATS].astype(float)
        if p.isna().any().any():
            raise SystemExit(f"Valeurs manquantes dans les profils :\n{p[p.isna().any(axis=1)]}\n"
                             "-> corrige step1 (proxy O*NET) au lieu de remplir par une valeur arbitraire.")
        print(f"[profils] O*NET + expert : {PROFILE_PATH}")
        return p, "onet_expert"
    print("[profils] career_profiles.csv absent -> profils de secours (lance step1 pour O*NET)")
    return pd.DataFrame(FALLBACK_PROFILES, index=FEATS).T, "fallback"


def generate(n, seed, profiles):
    rng = np.random.default_rng(seed)
    careers = list(profiles.index)
    P = profiles.values  # (C, F), 0-10

    # Répartition des métiers légèrement déséquilibrée (plus réaliste qu'uniforme)
    prior = rng.dirichlet(np.ones(len(careers)) * 8)
    y = rng.choice(len(careers), size=n, p=prior)

    # ---- aptitudes latentes (0-10)
    general = rng.normal(0, 1.2, size=(n, 1))              # "niveau" global de l'étudiant
    alpha = rng.uniform(0.45, 0.85, size=(n, 1))           # à quel point il colle au profil
    base = rng.uniform(2, 7, size=(n, P.shape[1]))         # compétences hors-profil
    latent = alpha * P[y] + (1 - alpha) * base + general + rng.normal(0, 1.0, size=(n, P.shape[1]))
    latent = np.clip(latent, 0, 10)

    df = pd.DataFrame({"student_id": [f"S{i:05d}" for i in range(1, n + 1)]})
    df["year_of_study"] = rng.integers(1, 6, size=n)
    df["gender"] = rng.choice(["F", "M"], size=n)  # descriptif; à exclure des features (équité)

    # ---- académique (observé)
    gpa = 10 + 0.55 * latent.mean(1) + 0.4 * general[:, 0] + rng.normal(0, 1.3, n)
    df["gpa_20"] = np.clip(gpa, 6, 20).round(2)
    df["grade_math_20"] = np.clip(8 + 1.0 * latent[:, FEATS.index("math_stats")] + rng.normal(0, 2, n), 0, 20).round(1)
    df["grade_programming_20"] = np.clip(8 + 1.0 * latent[:, FEATS.index("programming")] + rng.normal(0, 2, n), 0, 20).round(1)
    df["grade_networks_20"] = np.clip(8 + 1.0 * latent[:, FEATS.index("networking")] + rng.normal(0, 2, n), 0, 20).round(1)

    # ---- métier "rêvé" : une partie des étudiants est attirée par un autre métier que le leur
    # et SURESTIME les compétences de ce métier (ex. se note 9/10 en Kubernetes sans le maîtriser).
    # Seuls les Career Labs, qui mesurent les aptitudes réelles, peuvent révéler l'écart.
    # Générateur séparé : les autres tirages (et donc le reste du dataset) ne changent pas.
    rng_dream = np.random.default_rng(seed + 1)
    dreamer = rng_dream.random(n) < DREAMER_SHARE
    dream = (y + rng_dream.integers(1, len(careers), n)) % len(careers)  # toujours ≠ du vrai métier
    overrate = np.clip(P[dream] - P[y], 0, None) * rng_dream.uniform(0.6, 1.0, size=(n, 1))
    overrate *= dreamer[:, None]

    # ---- auto-évaluation (observée, biaisée : sur/sous-confiance individuelle + métier rêvé)
    confidence_bias = rng.normal(0, 1.2, size=(n, 1))
    self_eval = np.clip(np.round(latent + confidence_bias + rng.normal(0, 1.3, latent.shape) + overrate), 0, 10)
    for j, f in enumerate(FEATS):
        df[f"self_{f}"] = self_eval[:, j].astype(int)

    # ---- intérêts déclarés 1-5 (bruités ; 20% des étudiants ont un intérêt "trompeur",
    # et les étudiants "rêveurs" déclarent l'intérêt du métier rêvé)
    true_dom = np.array([INTEREST_DOMAINS.index(CAREER_INTEREST[careers[k]]) for k in y])
    misleading = rng.random(n) < 0.20
    shown_dom = np.where(misleading, rng.integers(0, len(INTEREST_DOMAINS), n), true_dom)
    dream_dom = np.array([INTEREST_DOMAINS.index(CAREER_INTEREST[careers[k]]) for k in dream])
    shown_dom = np.where(dreamer, dream_dom, shown_dom)
    for d_i, d in enumerate(INTEREST_DOMAINS):
        v = rng.normal(2.4, 0.9, n) + 1.8 * (shown_dom == d_i)
        df[f"interest_{d}"] = np.clip(np.round(v), 1, 5).astype(int)

    # ---- Career Labs (observés)
    labs_done = np.zeros(n, dtype=int)
    for lab, w in LABS.items():
        idx = [FEATS.index(f) for f in w]
        wv = np.array(list(w.values()))
        skill = latent[:, idx] @ wv                                   # 0-10
        affinity = np.isin(np.array(careers)[y], LAB_DOMAIN[lab]).astype(float)
        dream_affinity = dreamer & np.isin(np.array(careers)[dream], LAB_DOMAIN[lab])

        # probabilité de faire le lab : plus élevée si le domaine l'attire (vrai métier ou métier rêvé)
        p_take = 0.55 + 0.35 * np.maximum(affinity, dream_affinity)
        took = rng.random(n) < p_take
        labs_done += took

        # Le lab est une mesure plus fiable que l'auto-évaluation : bruit réduit (écart-type 5 au lieu de 7)
        score = 100 / (1 + np.exp(-(skill - 5.2) * 0.9)) + rng.normal(0, 7, n) * LAB_NOISE / 7
        score = np.clip(score, 0, 100)
        time_min = np.clip(rng.normal(75 - 4.5 * skill, 10, n), 10, 150)
        attempts = 1 + rng.poisson(np.clip(3.2 - 0.3 * skill, 0.1, None))
        hints = rng.poisson(np.clip(4 - 0.4 * skill, 0.05, None))
        completed = (score >= 50).astype(int)
        enjoy = np.clip(np.round(rng.normal(2.7 + 1.2 * affinity + 0.08 * (skill - 5), 0.8)), 1, 5)

        for col, val in [("score", score.round(1)), ("time_min", time_min.round(0)),
                         ("attempts", attempts), ("hints_used", hints),
                         ("completed", completed), ("enjoyment_1_5", enjoy)]:
            s = pd.Series(val, dtype=float)
            s[~took] = np.nan
            df[f"{lab}__{col}"] = s

    df["n_labs_done"] = labs_done

    # ---- LABELS
    df["career_label"] = np.array(careers)[y]
    # label "doux" : adéquation de l'étudiant à chaque métier (distance aptitudes latentes <-> profils)
    d = ((latent[:, None, :] - P[None, :, :]) ** 2).mean(-1)
    soft = np.exp(-d / 2.0)
    soft /= soft.sum(1, keepdims=True)
    for k, c in enumerate(careers):
        df[f"fit_{c}"] = soft[:, k].round(4)

    latent_df = pd.DataFrame(latent.round(2), columns=[f"latent_{f}" for f in FEATS])
    latent_df.insert(0, "student_id", df["student_id"])
    latent_df["career_label"] = df["career_label"]
    return df, latent_df


def data_dictionary(df):
    rows = []
    for c in df.columns:
        if c == "student_id": desc, role = "Identifiant anonyme", "id"
        elif c == "career_label": desc, role = "Métier cible (vérité terrain de la simulation)", "label"
        elif c.startswith("fit_"): desc, role = "Adéquation douce au métier (0-1, somme=1) — label alternatif", "soft_label"
        elif c.startswith("self_"): desc, role = "Auto-évaluation 0-10 de la compétence", "feature"
        elif c.startswith("interest_"): desc, role = "Intérêt déclaré 1-5 pour le domaine", "feature"
        elif "__" in c:
            lab, m = c.split("__"); desc, role = f"Career Lab {lab[4:]} : {m} (NaN = lab non fait)", "feature"
        elif c == "gender": desc, role = "Descriptif — NE PAS utiliser comme feature (audit d'équité)", "sensitive"
        else: desc, role = "Donnée académique / profil", "feature"
        rows.append({"column": c, "role": role, "dtype": str(df[c].dtype),
                     "missing_%": round(df[c].isna().mean() * 100, 1), "description": desc})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    profiles, src = load_profiles()
    df, latent = generate(args.n, args.seed, profiles)
    df.insert(1, "profile_source", src)

    df.to_csv(os.path.join(OUT_DIR, "careerverse_students.csv"), index=False)
    latent.to_csv(os.path.join(OUT_DIR, "careerverse_students_latent.csv"), index=False)
    data_dictionary(df).to_csv(os.path.join(OUT_DIR, "data_dictionary.csv"), index=False)

    print(f"[OK] {len(df)} étudiants, {df.shape[1]} colonnes")
    print(df["career_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
