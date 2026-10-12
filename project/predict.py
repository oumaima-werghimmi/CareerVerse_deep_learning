
"""
CareerVerse — Étape 6 : prédiction des métiers d'un étudiant.

Ce fichier permet de charger un modèle entraîné et de prédire
les métiers les plus adaptés au profil d'un étudiant.

Le programme reçoit les informations de l'étudiant au format JSON
et retourne les trois métiers les plus probables avec leurs probabilités.

Il vérifie également les données saisies et signale les erreurs
lorsque les informations sont invalides ou incomplètes.
"""

import argparse
import json
import math
import os

import numpy as np
import torch

from train import make_model

# Dossier contenant le modèle entraîné et les fichiers de préparation des données.
ART = "artifacts"

# Valeurs minimales et maximales autorisées pour les informations de l'étudiant.
BOUNDS = {"year_of_study": (1, 5), "gpa_20": (0, 20), "grade_math_20": (0, 20),
          "grade_programming_20": (0, 20), "grade_networks_20": (0, 20)}

# Valeurs autorisées pour les mesures réalisées dans les laboratoires.
LAB_BOUNDS = {"score": (0, 100), "time_min": (0, 600), "attempts": (1, 100), "hints_used": (0, 100),
              "enjoyment_1_5": (1, 5)}


# Classe utilisée pour signaler les erreurs dans les données de l'étudiant.
class InvalidStudent(ValueError):
    pass


# Vérifie qu'une valeur est numérique, finie et comprise dans les limites autorisées.
def _number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InvalidStudent(f"« {name} » doit être un nombre (reçu : {value!r}).")
    if not low <= value <= high:
        raise InvalidStudent(f"« {name} » doit être entre {low} et {high} (reçu : {value}).")
    return float(value)


# Détermine les limites autorisées selon le type de donnée.
def static_bounds(col):
    if col in BOUNDS:
        return BOUNDS[col]
    if col.startswith("self_"):
        return (0, 10)
    if col.startswith("interest_"):
        return (1, 5)
    return (-math.inf, math.inf)


# Classe principale qui charge le modèle et réalise les prédictions.
class CareerPredictor:

    # Charge le modèle entraîné et les informations nécessaires au prétraitement.
    def __init__(self, ckpt=f"{ART}/careernet.pt", meta=f"{ART}/preprocess.json"):
        state = torch.load(ckpt, map_location="cpu")

        # Vérifie si le checkpoint contient déjà les informations de prétraitement.
        if "meta" in state:
            self.meta = state["meta"]
            kind, kw = state["kind"], state["model_kwargs"]
        else:  # Utilise l'ancien format du checkpoint si nécessaire.
            with open(meta, encoding="utf-8") as f:
                self.meta = json.load(f)
            m = self.meta
            kind = state.get("kind", "careernet")
            kw = {"n_static": len(m["static_cols"]), "n_labs": len(m["labs"]),
                  "n_metrics": len(m["lab_metrics"]), "n_classes": len(m["classes"])}

        # Initialise le modèle et récupère les paramètres de prédiction.
        self.kind = kind
        self.temperature = float(state.get("temperature", 1.0))
        self.model = make_model(kind, kw)
        self.model.load_state_dict(state["state_dict"])

        # Place le modèle en mode évaluation.
        self.model.eval()

    # Prépare et transforme les données de l'étudiant avant la prédiction.
    def _encode(self, s):
        if not isinstance(s, dict):
            raise InvalidStudent("L'étudiant doit être un objet JSON.")

        # Initialise les données des laboratoires et le masque indiquant ceux qui sont terminés.
        m = self.meta
        lab = np.zeros((len(m["labs"]), len(m["lab_metrics"])))
        mask = np.zeros(len(m["labs"]), dtype=bool)

        # Parcourt les laboratoires pour vérifier leurs mesures.
        for i, l in enumerate(m["labs"]):
            vals = {k: s.get(f"{l}__{k}") for k in m["lab_metrics"]}
            given = [k for k, v in vals.items() if v is not None]

            # Si aucune mesure n'est fournie, le laboratoire est considéré comme non terminé.
            if not given:
                continue

            # Refuse un laboratoire dont certaines mesures sont manquantes.
            if len(given) < len(vals):
                missing = [f"{l}__{k}" for k, v in vals.items() if v is None]
                raise InvalidStudent(f"Lab {l} incomplet : il manque {', '.join(missing)}. "
                                     "Donnez les 5 mesures, ou aucune si le lab n'a pas été fait.")

            # Vérifie les mesures et applique la transformation log si nécessaire.
            v = np.array([_number(vals[k], f"{l}__{k}", *LAB_BOUNDS.get(k, (-math.inf, math.inf)))
                          for k in m["lab_metrics"]])
            for j, k in enumerate(m["lab_metrics"]):
                if k in m["log_metrics"]:
                    v[j] = np.log1p(v[j])

            # Normalise les mesures et indique que le laboratoire est terminé.
            lab[i] = (v - np.array(m["lab_mean"])) / np.array(m["lab_std"])
            mask[i] = True

        # Récupère et vérifie les informations générales de l'étudiant.
        static = []
        for c in m["static_cols"]:
            if c == "n_labs_done":
                static.append(float(mask.sum()))
                continue
            if c not in s or s[c] is None:
                raise InvalidStudent(f"Champ obligatoire manquant : « {c} ».")
            static.append(_number(s[c], c, *static_bounds(c)))

        # Normalise les informations générales pour les adapter au modèle.
        static = (np.array(static) - np.array(m["static_mean"])) / np.array(m["static_std"])

        # Convertit les données en tenseurs utilisables par PyTorch.
        t = lambda a, dt: torch.tensor(a[None], dtype=dt)
        return t(static, torch.float32), t(lab, torch.float32), t(mask, torch.bool)

    # Prédit les métiers les plus adaptés au profil de l'étudiant.
    @torch.no_grad()
    def predict(self, student, k=3):
        # Prépare les informations générales et les résultats des laboratoires.
        s, l, m = self._encode(student)

        # Calcule les probabilités associées à chaque métier.
        proba = torch.softmax(self.model(s, l, m) / self.temperature, -1)[0].numpy()

        # Sélectionne les k métiers ayant les probabilités les plus élevées.
        top = proba.argsort()[::-1][:k]

        # Retourne les métiers recommandés et le nombre de laboratoires terminés.
        return {"top_careers": [{"career": self.meta["classes"][i], "probability": round(float(proba[i]), 3)}
                                for i in top],
                "labs_done": int(m.sum())}


# Point d'entrée du programme lorsqu'il est exécuté depuis le terminal.
if __name__ == "__main__":

    # Définit les arguments que l'utilisateur peut fournir en ligne de commande.
    ap = argparse.ArgumentParser()
    ap.add_argument("student", nargs="?", default="example_student.json")
    ap.add_argument("--ckpt", default=os.path.join(ART, "careernet.pt"))
    a = ap.parse_args()

    # Charge les données de l'étudiant depuis le fichier JSON.
    with open(a.student, encoding="utf-8") as f:
        student = json.load(f)

    # Lance la prédiction et affiche le résultat au format JSON.
    try:
        print(json.dumps(CareerPredictor(a.ckpt).predict(student), indent=2, ensure_ascii=False))

    # Affiche un message clair si les données de l'étudiant sont invalides.
    except InvalidStudent as exc:
        raise SystemExit(f"Entrée invalide : {exc}")
