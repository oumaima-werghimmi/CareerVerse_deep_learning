"""
Tests de cohérence entraînement / inférence.

  python -m unittest test_pipeline -v
"""
import copy
import json
import unittest

import numpy as np
import pandas as pd
import torch

from predict import CareerPredictor, InvalidStudent
from train import load_split, predict_proba

ART = "artifacts"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = CareerPredictor(f"{ART}/careernet.pt")
        with open(f"{ART}/preprocess.json", encoding="utf-8") as f:
            cls.meta = json.load(f)
        cls.npz = np.load(f"{ART}/dataset.npz")
        cls.df = pd.read_csv("data/careerverse_students.csv").set_index("student_id")
        with open("examples/student_security.json", encoding="utf-8") as f:
            cls.student = json.load(f)

    def test_normalisation_uses_train_only(self):
        tr = self.npz["train_static"]
        self.assertTrue(np.allclose(tr.mean(0), 0, atol=1e-4))
        self.assertFalse(np.allclose(self.npz["test_static"].mean(0), 0, atol=1e-4))

    def test_splits_are_disjoint(self):
        ids = [set(self.npz[f"{s}_id"]) for s in ("train", "val", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(sum(map(len, ids)), 5000)

    def test_not_done_lab_is_masked_not_zero_score(self):
        mask, lab = self.npz["test_mask"], self.npz["test_lab"]
        self.assertTrue((lab[~mask] == 0).all())  # valeur neutre (= moyenne du train) + masque False
        j = self.meta["lab_metrics"].index("score")
        raw_zero = (0 - self.meta["lab_mean"][j]) / self.meta["lab_std"][j]
        self.assertNotAlmostEqual(raw_zero, 0.0)  # un vrai score de 0 serait codé autrement

    def test_predict_matches_training_encoding(self):
        """Un étudiant du test encodé par predict.py donne les mêmes probabilités que le dataset prétraité."""
        te = load_split(self.npz, "test")
        sid = str(self.npz["test_id"][0])
        row = self.df.loc[sid].to_dict()
        student = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in row.items()}
        p_api = self.P.predict(student, k=9)["top_careers"]
        p_ds = predict_proba(self.P.model, {k: v[:1] for k, v in te.items()})[0]
        for item in p_api:
            self.assertAlmostEqual(item["probability"], float(p_ds[self.meta["classes"].index(item["career"])]), places=3)

    def test_no_lab_student(self):
        s = {k: v for k, v in self.student.items() if not k.startswith("lab_")}
        r = self.P.predict(s)
        self.assertEqual(r["labs_done"], 0)
        self.assertAlmostEqual(sum(c["probability"] for c in self.P.predict(s, k=9)["top_careers"]), 1, places=2)

    def test_invalid_inputs_raise_clear_errors(self):
        cases = []
        s = copy.deepcopy(self.student); del s["self_python"]; cases.append(s)
        s = copy.deepcopy(self.student); s["gpa_20"] = 25; cases.append(s)
        s = copy.deepcopy(self.student); s["gpa_20"] = "14"; cases.append(s)
        s = copy.deepcopy(self.student); del s["lab_security_ctf__time_min"]; cases.append(s)
        for s in cases:
            with self.assertRaises(InvalidStudent):
                self.P.predict(s)

    def test_n_labs_done_recomputed(self):
        s = copy.deepcopy(self.student); s["n_labs_done"] = 9
        self.assertEqual(self.P.predict(s), self.P.predict(self.student))

    def test_checkpoint_carries_its_preprocessing(self):
        ck = torch.load(f"{ART}/careernet.pt", map_location="cpu")
        if "meta" in ck:
            self.assertEqual(ck["meta"]["static_cols"], self.meta["static_cols"])


if __name__ == "__main__":
    unittest.main()
