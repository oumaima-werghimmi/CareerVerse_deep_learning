"""
CareerVerse — Étape 1 : collecte des connaissances métiers depuis O*NET
=======================================================================

Télécharge la base O*NET (fichiers texte, domaine public, U.S. Dept. of Labor),
garde uniquement les métiers IT ciblés par CareerVerse et construit, pour chaque
métier, un PROFIL de compétences sur une échelle 0-10.

Sortie :
  data/onet_raw/...                 fichiers O*NET bruts
  data/career_profiles.csv          1 ligne par métier, 1 colonne par compétence (0-10)
  data/career_skills_long.csv       format long (métier, source, élément, importance)
  data/career_technologies.csv      technologies citées par O*NET pour chaque métier

Usage :
  pip install pandas requests
  python step1_collect_onet.py
  # ou, si tu as déjà téléchargé le zip à la main depuis https://www.onetcenter.org/database.html :
  python step1_collect_onet.py --zip chemin/vers/db_XX_X_text.zip
"""
import argparse
import io
import os
import zipfile

import pandas as pd

OUT_DIR = "data"
RAW_DIR = os.path.join(OUT_DIR, "onet_raw")

# Versions essayées dans l'ordre (la plus récente d'abord). Ajoute-en si une nouvelle sort.
ONET_VERSIONS = ["30_1", "30_0", "29_3", "29_2", "29_1", "29_0"]
ONET_URL = "https://www.onetcenter.org/dl_files/database/db_{v}_text.zip"

# ---------------------------------------------------------------------------
# 1. Métiers CareerVerse  ->  codes O*NET-SOC
# ---------------------------------------------------------------------------
CAREERS = {
    "Backend Developer":      ["15-1252.00"],               # Software Developers
    "Frontend/Web Developer": ["15-1254.00", "15-1255.00"], # Web Developers + Web & Digital Interface Designers
    "DevOps Engineer":        ["15-1299.08"],               # Computer Systems Engineers/Architects
    "Cloud Engineer":         ["15-1241.00", "15-1299.08"], # Computer Network Architects (+ systems eng.)
    "Cybersecurity Analyst":  ["15-1212.00"],               # Information Security Analysts
    "Data Scientist":         ["15-2051.00"],               # Data Scientists
    "Database Administrator": ["15-1242.00"],               # Database Administrators
    "System/Network Admin":   ["15-1244.00"],               # Network and Computer Systems Administrators
    "QA/Test Engineer":       ["15-1253.00"],               # Software Quality Assurance Analysts and Testers
}

# Métiers proches utilisés quand O*NET n'a pas de données Skills/Knowledge/Abilities
KSA_PROXY = {
    "Data Scientist": ["15-2041.00", "15-1221.00"],  # Statisticians + Computer & Information Research Scientists
}
LOW = 1.0  # note minimale d'un profil (tout métier IT demande un minimum de chaque compétence)
W_ONET = 0.5  # poids du signal O*NET dans le profil final (le reste = profil expert)

# Profil expert (0-10), ordre des colonnes = FEATURES
EXPERT_PRIOR = {
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
# Compétence "signature" de chaque métier (contrôle de cohérence)
SIGNATURE = {
    "Backend Developer": "programming", "Frontend/Web Developer": "web_frontend",
    "DevOps Engineer": "docker_k8s", "Cloud Engineer": "cloud",
    "Cybersecurity Analyst": "security", "Data Scientist": "math_stats",
    "Database Administrator": "databases_sql", "System/Network Admin": "networking",
    "QA/Test Engineer": "testing_quality",
}

# ---------------------------------------------------------------------------
# 2. Compétences CareerVerse (= features du modèle) et leur ancrage O*NET
#    - "onet": éléments des fichiers Skills.txt / Knowledge.txt / Abilities.txt
#    - "tech": mots-clés cherchés dans Technology Skills.txt
# ---------------------------------------------------------------------------
FEATURES = {
    "programming":     {"onet": ["Programming"],
                        "tech": ["python", "java", "c#", "c++", "javascript", "go", "ruby", "node.js"]},
    "python":          {"onet": [], "tech": ["python"]},
    "linux":           {"onet": [], "tech": ["linux", "unix", "bash", "shell", "red hat", "ubuntu"]},
    "docker_k8s":      {"onet": [], "tech": ["docker", "kubernetes", "ansible", "terraform", "jenkins", "puppet", "chef"]},
    "cloud":           {"onet": [], "tech": ["amazon web services", "aws", "azure", "google cloud", "cloud"]},
    "networking":      {"onet": ["Telecommunications"],
                        "tech": ["cisco", "network", "tcp/ip", "wireshark", "vpn", "firewall"]},
    "security":        {"onet": ["Public Safety and Security"],
                        "tech": ["security", "intrusion", "encryption", "nessus", "metasploit", "splunk", "firewall", "antivirus"]},
    "databases_sql":   {"onet": [],
                        "tech": ["sql", "oracle", "mysql", "postgresql", "mongodb", "database", "db2", "nosql"]},
    "web_frontend":    {"onet": ["Design"],
                        "tech": ["html", "css", "javascript", "react", "angular", "vue", "typescript", "jquery", "adobe"]},
    "math_stats":      {"onet": ["Mathematics"],
                        "tech": ["r", "sas", "matlab", "tensorflow", "pytorch", "scikit", "spss", "tableau"]},
    "testing_quality": {"onet": ["Quality Control Analysis", "Operations Monitoring"],
                        "tech": ["selenium", "junit", "test", "jira", "cucumber"]},
    "problem_solving": {"onet": ["Complex Problem Solving", "Critical Thinking", "Deductive Reasoning"], "tech": []},
    "systems_thinking":{"onet": ["Systems Analysis", "Systems Evaluation", "Troubleshooting"], "tech": []},
}


def download_onet(zip_path=None):
    """Retourne un objet ZipFile O*NET (local ou téléchargé)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    if zip_path:
        print(f"[O*NET] Lecture du zip local {zip_path}")
        return zipfile.ZipFile(zip_path)
    import requests
    for v in ONET_VERSIONS:
        url = ONET_URL.format(v=v)
        print(f"[O*NET] Essai {url} ...")
        try:
            r = requests.get(url, timeout=120)
        except requests.RequestException as e:
            print("   échec :", e)
            continue
        if r.status_code == 200 and r.content[:2] == b"PK":
            local = os.path.join(RAW_DIR, f"db_{v}_text.zip")
            with open(local, "wb") as f:
                f.write(r.content)
            print(f"   OK -> {local} ({len(r.content)/1e6:.1f} Mo)")
            return zipfile.ZipFile(io.BytesIO(r.content))
        print("   HTTP", r.status_code)
    raise SystemExit("Impossible de télécharger O*NET. Télécharge le zip 'Text' sur "
                     "https://www.onetcenter.org/database.html puis relance avec --zip.")


def read_table(zf, name):
    member = next(m for m in zf.namelist() if m.endswith("/" + name) or m == name)
    with zf.open(member) as f:
        return pd.read_csv(f, sep="\t", encoding="utf-8", dtype=str)


def scaled_importance(df, soc_codes, names):
    """Moyenne des scores d'importance (Scale IM, 1-5) -> 0-10."""
    sub = df[(df["O*NET-SOC Code"].isin(soc_codes)) &
             (df["Element Name"].isin(names)) &
             (df["Scale ID"] == "IM")]
    if sub.empty:
        return None
    im = sub["Data Value"].astype(float).mean()
    return (im - 1) / 4 * 10


def tech_share(tech, soc_codes, keywords):
    """Part des technologies du métier qui correspondent aux mots-clés (0-1)."""
    sub = tech[tech["O*NET-SOC Code"].isin(soc_codes)]
    if sub.empty or not keywords:
        return None
    text = (sub["Example"].fillna("") + " | " + sub["Commodity Title"].fillna("")).str.lower()
    hits = 0
    for t in text:
        for k in keywords:
            # mots courts (ex. "r", "go") : correspondance de mot entier
            if len(k) <= 2:
                if f" {k} " in f" {t.replace(',', ' ').replace('(', ' ').replace(')', ' ')} ":
                    hits += 1
                    break
            elif k in t:
                hits += 1
                break
    return hits / len(sub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", help="chemin vers un zip O*NET déjà téléchargé")
    args = ap.parse_args()

    zf = download_onet(args.zip)
    occ = read_table(zf, "Occupation Data.txt")
    skills = read_table(zf, "Skills.txt")
    knowledge = read_table(zf, "Knowledge.txt")
    abilities = read_table(zf, "Abilities.txt")
    tech = read_table(zf, "Technology Skills.txt")
    ksa = pd.concat([skills, knowledge, abilities], ignore_index=True)

    all_socs = sorted({s for v in CAREERS.values() for s in v})
    missing = set(all_socs) - set(occ["O*NET-SOC Code"])
    if missing:
        print("[!] Codes absents de cette version O*NET :", missing)

    # --- format long : tout ce qu'O*NET dit sur nos métiers (utile pour le rapport / graph)
    long_rows = []
    for career, socs in CAREERS.items():
        sub = ksa[ksa["O*NET-SOC Code"].isin(socs) & (ksa["Scale ID"] == "IM")]
        g = sub.groupby("Element Name")["Data Value"].apply(lambda s: s.astype(float).mean())
        for elem, val in g.items():
            long_rows.append({"career": career, "element": elem, "importance_1_5": round(val, 2)})
    pd.DataFrame(long_rows).to_csv(os.path.join(OUT_DIR, "career_skills_long.csv"), index=False)

    t = tech[tech["O*NET-SOC Code"].isin(all_socs)].copy()
    soc2career = {}
    for c, socs in CAREERS.items():
        for s in socs:
            soc2career.setdefault(s, []).append(c)
    t["careers"] = t["O*NET-SOC Code"].map(lambda s: "; ".join(soc2career.get(s, [])))
    t.to_csv(os.path.join(OUT_DIR, "career_technologies.csv"), index=False)

    # --- profils 0-10 par métier  (v2 : corrige 2 biais de la v1)
    # (a) Certains métiers n'ont pas de données Skills/Knowledge/Abilities dans O*NET
    #     (ex. 15-2051.00 Data Scientists). On utilise alors des métiers proches (KSA_PROXY).
    # (b) Les listes de technologies n'ont pas la même densité selon les métiers
    #     (87 lignes pour Data Scientist contre ~250-430 ailleurs). La v1 utilisait la part brute,
    #     ce qui mettait Data Scientist à 10 partout. On mesure maintenant l'emphase RELATIVE :
    #     part de la feature / part moyenne du métier sur toutes les features tech.
    ksa_socs = {c: [s for s in socs if (ksa["O*NET-SOC Code"] == s).any()] or KSA_PROXY.get(c, [])
                for c, socs in CAREERS.items()}
    for c in CAREERS:
        if ksa_socs[c] != [s for s in CAREERS[c] if (ksa["O*NET-SOC Code"] == s).any()]:
            print(f"[proxy] {c}: pas de KSA O*NET -> utilise {ksa_socs[c]}")

    ksa_raw = pd.DataFrame({c: {f: (scaled_importance(ksa, ksa_socs[c], spec["onet"]) if spec["onet"] else None)
                                 for f, spec in FEATURES.items()} for c in CAREERS}).T.astype(float)
    tech_raw = pd.DataFrame({c: {f: (tech_share(tech, socs, spec["tech"]) if spec["tech"] else None)
                                  for f, spec in FEATURES.items()} for c, socs in CAREERS.items()}).T.astype(float)
    tech_rel = tech_raw.div(tech_raw.mean(axis=1), axis=0)   # emphase relative dans le métier

    def minmax(df, lo=LOW, hi=10.0):
        """Normalise chaque feature entre métiers : le métier le plus concerné = 10."""
        mn, mx = df.min(), df.max()
        return lo + (df - mn) / (mx - mn).replace(0, 1) * (hi - lo)

    parts = [minmax(ksa_raw), minmax(tech_rel)]
    onet_prof = pd.concat(parts).groupby(level=0).mean().reindex(list(CAREERS))[list(FEATURES)]
    onet_prof.round(2).to_csv(os.path.join(OUT_DIR, "career_profiles_onet_only.csv"))

    # (c) Le signal O*NET seul reste bruité pour distinguer finement des métiers IT (mots-clés).
    #     Profil final = W_ONET * O*NET + (1 - W_ONET) * profil expert. Documenté dans le rapport.
    expert = pd.DataFrame(EXPERT_PRIOR, index=list(FEATURES)).T.reindex(list(CAREERS))
    prof = (W_ONET * onet_prof + (1 - W_ONET) * expert).round(2)

    # contrôle de cohérence : chaque métier doit être n°1 (ou n°2) sur sa compétence signature
    print("\n[contrôle] rang du métier sur sa compétence signature :")
    for c, f in SIGNATURE.items():
        rank = int(prof[f].rank(ascending=False)[c])
        print(f"   {'OK ' if rank <= 2 else '!! '} {c:24s} {f:16s} rang {rank}")
    prof.index.name = "career"
    prof.insert(0, "onet_soc_codes", [", ".join(CAREERS[c]) for c in prof.index])
    prof.to_csv(os.path.join(OUT_DIR, "career_profiles.csv"))
    print("\n[OK] Profils métiers (0-10) :\n")
    print(prof.drop(columns="onet_soc_codes").round(1).to_string())
    print(f"\nFichiers écrits dans {OUT_DIR}/")


if __name__ == "__main__":
    main()
