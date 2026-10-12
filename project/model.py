"""
CareerNet — architecture deep learning de CareerVerse
======================================================

                 ┌──────────────────────────────┐
  profil (24) ──►│ MLP 24→128→64 (GELU, Dropout)│──► h_profil (64)
                 └──────────────────────────────┘                      ┌──────────────┐
                                                           concat ───►│ MLP 128→64→9 │──► logits (9 métiers)
  labs (9×5) ─► Linear 5→64 + embedding du lab ─┐                      └──────────────┘
  masque (9) ───────────────────────────────────┤
                 [CLS] + 9 tokens de labs ──► Transformer encoder (2 couches, 4 têtes)
                                           ──► sortie du token CLS = h_labs (64)

Pourquoi un Transformer sur les labs ?
  - chaque étudiant fait un sous-ensemble différent de labs : le masque d'attention
    ignore proprement les labs non faits (pas d'imputation artificielle) ;
  - l'attention apprend les interactions entre labs (ex. bon en CTF + bon en réseau -> Cyber) ;
  - les poids d'attention du CLS indiquent quels labs ont compté -> explicabilité ;
  - un nouveau lab = un nouveau token : l'architecture s'étend sans tout redessiner.

Variantes pour les expériences de fusion (fusion_experiments.py) :
  - ProfileOnly : branche profil seule (sans n_labs_done, qui est une information de labs) ;
  - LabsOnly    : branche labs seule (labs + masque) ;
  - GatedCareerNet : comme CareerNet, mais une porte apprise pondère, dimension par dimension,
    la part du profil et celle des labs selon les deux représentations et la fraction de labs faits.
`use_metrics` permet de neutraliser une mesure de lab (ablation) sans changer le state_dict.
"""

import torch
import torch.nn as nn


# Encode les résultats des Career Labs pour obtenir une représentation globale.
class LabEncoder(nn.Module):
    """Transforme les résultats des labs en une représentation utilisable par le modèle."""

    def __init__(self, n_labs, n_metrics, d=64, heads=4, layers=2, dropout=0.2):
        super().__init__()

        # Transforme les mesures de chaque lab en vecteurs de dimension d.
        self.lab_proj = nn.Linear(n_metrics, d)

        # Ajoute une représentation permettant d'identifier chaque lab.
        self.lab_emb = nn.Embedding(n_labs, d)

        # Crée un token spécial CLS qui résume les informations des labs.
        self.cls = nn.Parameter(torch.zeros(1, 1, d))

        # Définit une couche Transformer pour analyser les relations entre les labs.
        enc_layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                               batch_first=True, activation="gelu", norm_first=True)

        # Crée le Transformer composé de plusieurs couches.
        self.encoder = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)

        # Enregistre les identifiants des labs.
        self.register_buffer("lab_ids", torch.arange(n_labs))

    # Traite les résultats des labs et retourne leur représentation globale.
    def forward(self, lab, mask):
        B = lab.size(0)

        # Transforme les mesures et ajoute l'identité de chaque lab.
        tokens = self.lab_proj(lab) + self.lab_emb(self.lab_ids)[None]

        # Ajoute le token CLS au début de la séquence.
        tokens = torch.cat([self.cls.expand(B, -1, -1), tokens], 1)

        # Ignore les labs non terminés pendant le traitement par le Transformer.
        # Le token CLS reste toujours actif.
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask], 1)

        # Retourne le vecteur CLS qui résume les informations des labs.
        return self.encoder(tokens, src_key_padding_mask=pad)[:, 0]


# Définit les mesures que le modèle doit utiliser.
def _metric_mask(module, n_metrics, use_metrics):
    # Utilise toutes les mesures si aucune sélection n'est fournie.
    keep = torch.ones(n_metrics) if use_metrics is None else torch.tensor(use_metrics, dtype=torch.float32)

    # Enregistre le masque sans l'ajouter aux poids sauvegardés du modèle.
    module.register_buffer("metric_mask", keep, persistent=False)


# Modèle principal qui combine le profil de l'étudiant et ses résultats des labs.
class CareerNet(nn.Module):
    def __init__(self, n_static, n_labs, n_metrics, n_classes, d=64, heads=4, layers=2, dropout=0.2,
                 use_metrics=None):
        super().__init__()

        # Réseau qui transforme les informations générales du profil.
        self.static_net = nn.Sequential(
            nn.Linear(n_static, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, d), nn.GELU(),
        )

        # Transforme les mesures des labs en vecteurs.
        self.lab_proj = nn.Linear(n_metrics, d)

        # Identifie chaque lab dans la séquence.
        self.lab_emb = nn.Embedding(n_labs, d)

        # Token utilisé pour résumer les informations des labs.
        self.cls = nn.Parameter(torch.zeros(1, 1, d))

        # Définit les couches du Transformer.
        enc_layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                               batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers, enable_nested_tensor=False)

        # Réseau final qui combine les deux représentations et prédit les métiers.
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_classes),
        )

        # Enregistre les identifiants des labs et les mesures à utiliser.
        self.register_buffer("lab_ids", torch.arange(n_labs))
        _metric_mask(self, n_metrics, use_metrics)

    # Effectue la prédiction à partir du profil et des labs.
    def forward(self, static, lab, mask):
        """static (B,S)  lab (B,L,M)  mask (B,L) bool, True = lab fait"""
        B = static.size(0)

        # Conserve uniquement les mesures sélectionnées.
        lab = lab * self.metric_mask

        # Encode les informations générales de l'étudiant.
        h_static = self.static_net(static)

        # Transforme les résultats des labs en tokens.
        tokens = self.lab_proj(lab) + self.lab_emb(self.lab_ids)[None]

        # Ajoute le token CLS pour résumer les informations.
        tokens = torch.cat([self.cls.expand(B, -1, -1), tokens], 1)

        # Ignore les labs non terminés pendant le traitement.
        # Le token CLS n'est jamais ignoré.
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), ~mask], 1)

        # Récupère la représentation globale des résultats des labs.
        h_labs = self.encoder(tokens, src_key_padding_mask=pad)[:, 0]

        # Combine les informations du profil et des labs pour prédire les métiers.
        return self.head(torch.cat([h_static, h_labs], 1))


# Modèle qui ajuste automatiquement l'importance du profil et des labs.
class GatedCareerNet(nn.Module):
    """Fusionne les informations du profil et des labs grâce à une porte apprise."""

    def __init__(self, n_static, n_labs, n_metrics, n_classes, d=64, heads=4, layers=2, dropout=0.2,
                 use_metrics=None):
        super().__init__()

        # Encode les informations générales du profil.
        self.static_net = nn.Sequential(
            nn.Linear(n_static, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, d), nn.GELU(),
        )

        # Encode les résultats des labs avec le Transformer.
        self.labs = LabEncoder(n_labs, n_metrics, d, heads, layers, dropout)

        # Calcule une porte qui contrôle le poids attribué aux deux représentations.
        self.gate = nn.Sequential(nn.Linear(2 * d + 1, d), nn.Sigmoid())

        # Combine les représentations pondérées pour prédire les métiers.
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_classes),
        )

        # Définit les mesures des labs à utiliser.
        _metric_mask(self, n_metrics, use_metrics)

    # Calcule les représentations du profil et des labs, ainsi que les poids de fusion.
    def gate_values(self, static, lab, mask):
        h_s = self.static_net(static)
        h_l = self.labs(lab * self.metric_mask, mask)

        # Calcule la proportion de labs terminés par l'étudiant.
        frac = mask.float().mean(1, keepdim=True)

        # Calcule les poids attribués au profil et aux labs.
        return self.gate(torch.cat([h_s, h_l, frac], 1)), h_s, h_l

    # Combine les deux sources d'informations avant la classification.
    def forward(self, static, lab, mask):
        g, h_s, h_l = self.gate_values(static, lab, mask)

        # Applique les poids calculés puis prédit la catégorie professionnelle.
        return self.head(torch.cat([g * h_s, (1 - g) * h_l], 1))


# Modèle de référence qui utilise uniquement les informations du profil.
class ProfileOnly(nn.Module):
    """Effectue les prédictions sans utiliser directement les résultats des labs."""

    def __init__(self, n_static, n_labs, n_metrics, n_classes, d=64, dropout=0.2, static_keep=None, **_):
        super().__init__()

        # Définit les informations du profil à conserver.
        keep = torch.ones(n_static) if static_keep is None else torch.tensor(static_keep, dtype=torch.float32)
        self.register_buffer("static_keep", keep, persistent=False)

        # Réseau qui prédit les métiers à partir du profil uniquement.
        self.net = nn.Sequential(
            nn.Linear(n_static, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_classes),
        )

    # Retourne les prédictions à partir des informations du profil.
    def forward(self, static, lab, mask):
        return self.net(static * self.static_keep)


# Modèle de référence qui utilise uniquement les résultats des Career Labs.
class LabsOnly(nn.Module):
    """Prédit les métiers à partir des résultats des labs uniquement."""

    def __init__(self, n_static, n_labs, n_metrics, n_classes, d=64, heads=4, layers=2, dropout=0.2,
                 use_metrics=None, **_):
        super().__init__()

        # Encode les résultats des labs.
        self.labs = LabEncoder(n_labs, n_metrics, d, heads, layers, dropout)

        # Transforme la représentation obtenue en prédictions de métiers.
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d, n_classes))

        # Définit les mesures à utiliser.
        _metric_mask(self, n_metrics, use_metrics)

    # Effectue la prédiction à partir des labs uniquement.
    def forward(self, static, lab, mask):
        return self.head(self.labs(lab * self.metric_mask, mask))


# Modèle de référence classique qui utilise un réseau de neurones MLP.
class MLPBaseline(nn.Module):
    """Utilise les informations du profil, les mesures des labs et leur masque."""

    def __init__(self, n_in, n_classes, dropout=0.2):
        super().__init__()

        # Définit les couches du réseau chargé de prédire les métiers.
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    # Prépare les données et calcule les prédictions.
    def forward(self, static, lab, mask):

        # Regroupe le profil, les mesures aplaties des labs et leur masque.
        x = torch.cat([static, lab.flatten(1), mask.float()], 1)

        # Retourne les scores prédits pour les différentes catégories professionnelles.
        return self.net(x)
