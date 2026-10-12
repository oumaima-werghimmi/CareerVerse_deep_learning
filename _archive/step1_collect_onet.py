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

    # --- profils 0-10 par métier
    raw = {}
    for career, socs in CAREERS.items():
        row = {}
        for feat, spec in FEATURES.items():
            o = scaled_importance(ksa, socs, spec["onet"]) if spec["onet"] else None
            s = tech_share(tech, socs, spec["tech"]) if spec["tech"] else None
            row[feat] = (o, s)
        raw[career] = row

    # la part "tech" est normalisée entre métiers (max = 10) pour chaque feature
    profiles = {}
    for feat in FEATURES:
        shares = [raw[c][feat][1] for c in CAREERS if raw[c][feat][1] is not None]
        mx = max(shares) if shares and max(shares) > 0 else 1
        for c in CAREERS:
            o, s = raw[c][feat]
            parts = []
            if o is not None:
                parts.append(o)
            if s is not None:
                parts.append(s / mx * 10)
            profiles.setdefault(c, {})[feat] = round(sum(parts) / len(parts), 2) if parts else None

    prof = pd.DataFrame(profiles).T
    prof.index.name = "career"
    prof.insert(0, "onet_soc_codes", [", ".join(CAREERS[c]) for c in prof.index])
    prof.to_csv(os.path.join(OUT_DIR, "career_profiles.csv"))
    print("\n[OK] Profils métiers (0-10) :\n")
    print(prof.drop(columns="onet_soc_codes").round(1).to_string())
    print(f"\nFichiers écrits dans {OUT_DIR}/")


if __name__ == "__main__":
    main()
