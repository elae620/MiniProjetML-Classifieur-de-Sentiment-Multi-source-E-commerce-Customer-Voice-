"""
train.py — Script d'entraînement reproductible du classifieur de sentiment.

Auteur  : [Votre Nom]
Projet  : Mini-Projet ML Appliqué 2026 — Classifieur Sentiment Multi-source
Exécution :
    python train.py

Sorties :
    models/model.pkl        → Pipeline scikit-learn sérialisé
    models/vectorizer.pkl   → TfidfVectorizer sérialisé (aussi inclus dans le pipeline)
    figures/distribution_classes.png
    figures/longueur_textes.png
    figures/matrice_confusion.png
"""

from __future__ import annotations

import os
import re
import warnings
import logging
from pathlib import Path
from typing import Tuple

import joblib
import matplotlib
matplotlib.use("Agg")          # Pas d'affichage interactif (compatible CI/server)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ── Réduction du bruit de logs ───────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
DATA_PATH    = Path("data/dataset_sentiment_ecommerce/textes.csv")
MODEL_DIR    = Path("models")
FIGURES_DIR  = Path("figures")
LABELS       = ["positif", "negatif", "neutre"]   # ordre stable pour les graphiques

# ── Stop-words français embarqués (pas de dépendance réseau NLTK) ─────────────
# Source : liste NLTK french stopwords — négations exclues volontairement
_SW_FR_BASE: frozenset[str] = frozenset({
    "a", "au", "aux", "avec", "ce", "ces", "cet", "cette",
    "dans", "de", "des", "du", "elle", "elles", "en", "et",
    "eu", "eux", "eu", "ici", "il", "ils", "je", "la", "le",
    "les", "leur", "leurs", "lui", "ma", "mais", "me", "même",
    "mes", "mon", "notre", "nous", "on", "ou", "où", "par",
    "pour", "que", "qui", "sa", "se", "ses", "si", "son",
    "sur", "ta", "te", "tes", "ton", "tu", "un", "une",
    "vos", "votre", "vous", "y", "été", "être", "avoir",
    "qu", "car", "donc", "or", "ni", "or", "les", "lors",
    "ainsi", "alors", "après", "aussi", "avant", "ça", "car",
    "cela", "ci", "comme", "d", "déjà", "depuis", "dès",
    "dont", "encore", "ensuite", "entre", "ét", "eu",
    "jusque", "là", "lorsque", "lui", "m", "malgré",
    "même", "moins", "n", "nôtre", "o", "par", "parce",
    "parfois", "plus", "pourquoi", "premier", "puis",
    "puisque", "quand", "quant", "quel", "quelle", "quelque",
    "quels", "quelles", "quoi", "s", "sans", "seul",
    "seulement", "sinon", "soit", "sous", "sujet",
    "tandis", "tant", "tous", "tout", "toute", "toutes",
    "très", "trop", "une", "vers", "via", "voilà",
    "while", "bien", "c", "ceux", "chaque",
    "chacun", "chacune", "chose", "contre", "d'une",
    "façon", "fois", "j", "jusqu", "l", "lequel",
    "lorsque", "malgré", "même", "moi", "moment",
    "personne", "peu", "peut", "peux", "plusieurs", "point",
    "premier", "quant", "quelques", "quoi", "rien",
    "toujours", "tout", "très",
})

# Négations françaises à CONSERVER — discriminantes pour le sentiment
NEGATIONS_FR: frozenset[str] = frozenset({
    "ne", "pas", "jamais", "plus", "guère", "nullement",
    "aucun", "aucune",
})

# Stop-words finaux : base MINUS négations
STOP_WORDS_FR: frozenset[str] = _SW_FR_BASE - NEGATIONS_FR


def _charger_stopwords_nltk() -> frozenset[str]:
    """
    Essaie de charger les stop-words NLTK (si disponibles).
    Retourne la liste embarquée en cas d'échec réseau ou de données manquantes.
    """
    try:
        from nltk.corpus import stopwords as sw_corpus
        sw = frozenset(sw_corpus.words("french")) - NEGATIONS_FR
        log.info("Stop-words NLTK chargés : %d mots", len(sw))
        return sw
    except Exception:
        pass
    try:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords as sw_corpus
        sw = frozenset(sw_corpus.words("french")) - NEGATIONS_FR
        log.info("Stop-words NLTK téléchargés : %d mots", len(sw))
        return sw
    except Exception:
        log.warning("NLTK indisponible — utilisation de la liste embarquée (%d mots)", len(STOP_WORDS_FR))
        return STOP_WORDS_FR


# Charge les meilleurs stop-words disponibles au démarrage
_STOP_WORDS_ACTIFS: frozenset[str] = _charger_stopwords_nltk()


# ── Nettoyage textuel ─────────────────────────────────────────────────────────
def nettoyer_texte(texte: str) -> str:
    """
    Nettoie un texte français pour la classification de sentiment.

    Transformations (dans l'ordre) :
    1. Mise en minuscules
    2. Suppression des URLs
    3. Suppression de la ponctuation (sauf apostrophes)
    4. Suppression des chiffres isolés
    5. Suppression des stop-words (négations préservées)
    6. Normalisation des espaces

    Args:
        texte: Chaîne brute issue du dataset.

    Returns:
        Chaîne nettoyée prête pour la vectorisation TF-IDF.
    """
    if not isinstance(texte, str):
        return ""

    # 1. Minuscules
    texte = texte.lower()

    # 2. URLs
    texte = re.sub(r"https?://\S+|www\.\S+", " ", texte)

    # 3. Ponctuation (on garde ' pour les contractions : ne → n')
    texte = re.sub(r"[^\w\s']", " ", texte)

    # 4. Chiffres isolés
    texte = re.sub(r"\b\d+\b", " ", texte)

    # 5. Stop-words (négations conservées)
    tokens = texte.split()
    tokens = [t for t in tokens if t not in _STOP_WORDS_ACTIFS]

    # 6. Espaces
    return " ".join(tokens).strip()


# ── Chargement & EDA ──────────────────────────────────────────────────────────
def charger_dataset(chemin: Path) -> pd.DataFrame:
    """
    Charge textes.csv, vérifie l'intégrité, ajoute les colonnes de nettoyage.

    Args:
        chemin: Chemin vers le fichier CSV.

    Returns:
        DataFrame avec colonnes [id, texte, sentiment_ref, texte_nettoye,
        nb_mots_brut, nb_mots_net].
    """
    df = pd.read_csv(chemin, encoding="utf-8")
    log.info("Dataset chargé : %d lignes, colonnes=%s", len(df), list(df.columns))

    # Vérifications
    assert df["texte"].notna().all(), "Valeurs manquantes dans 'texte'"
    assert df["sentiment_ref"].notna().all(), "Valeurs manquantes dans 'sentiment_ref'"

    n_doublons = df["texte"].duplicated().sum()
    if n_doublons:
        log.warning("%d doublon(s) supprimé(s)", n_doublons)
        df = df.drop_duplicates(subset="texte").reset_index(drop=True)

    df["texte_nettoye"] = df["texte"].apply(nettoyer_texte)
    df["nb_mots_brut"]  = df["texte"].str.split().str.len()
    df["nb_mots_net"]   = df["texte_nettoye"].str.split().str.len()

    return df


def eda_et_graphiques(df: pd.DataFrame, figures_dir: Path) -> None:
    """
    Génère et sauvegarde les figures EDA (distribution classes, longueur textes).

    Args:
        df          : DataFrame nettoyé.
        figures_dir : Dossier de sortie.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    palette = {"positif": "#2ecc71", "negatif": "#e74c3c", "neutre": "#3498db"}

    # ── Figure 1 : Distribution des classes ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["sentiment_ref"].value_counts().reindex(LABELS).fillna(0)
    bars = ax.bar(
        counts.index, counts.values,
        color=[palette[c] for c in counts.index],
        edgecolor="white", linewidth=1.2,
    )
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(int(val)), ha="center", va="bottom",
            fontweight="bold", fontsize=12,
        )
    ax.set_title("Distribution des classes de sentiment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment", fontsize=11)
    ax.set_ylabel("Nombre d'exemples", fontsize=11)
    ax.set_ylim(0, counts.max() + 2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(figures_dir / "distribution_classes.png", dpi=150)
    plt.close(fig)
    log.info("Figure sauvegardée : distribution_classes.png")

    # ── Figure 2 : Longueur des textes par classe ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    data_box = [df[df["sentiment_ref"] == c]["nb_mots_brut"].values for c in LABELS]
    bp = axes[0].boxplot(
        data_box, labels=LABELS, patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, c in zip(bp["boxes"], LABELS):
        patch.set_facecolor(palette[c])
        patch.set_alpha(0.7)
    axes[0].set_title("Longueur (mots) par classe — texte brut", fontsize=11)
    axes[0].set_ylabel("Nombre de mots")
    axes[0].spines[["top", "right"]].set_visible(False)

    for c in LABELS:
        subset = df[df["sentiment_ref"] == c]["nb_mots_brut"]
        axes[1].hist(subset, bins=range(5, 20), alpha=0.6,
                     label=c, color=palette[c], edgecolor="white")
    axes[1].set_title("Distribution de la longueur (toutes classes)", fontsize=11)
    axes[1].set_xlabel("Nombre de mots")
    axes[1].set_ylabel("Fréquence")
    axes[1].legend()
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.suptitle("EDA — Longueur des avis clients (texte brut)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(figures_dir / "longueur_textes.png", dpi=150)
    plt.close(fig)
    log.info("Figure sauvegardée : longueur_textes.png")


# ── Construction des pipelines ────────────────────────────────────────────────
def _make_tfidf() -> TfidfVectorizer:
    """Crée un TfidfVectorizer standard pour le projet."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
        stop_words=list(_STOP_WORDS_ACTIFS),
        analyzer="word",
    )


def construire_pipelines() -> dict[str, Pipeline]:
    """
    Crée les deux pipelines comparatifs :
    - 'lr'  : TF-IDF + Régression Logistique (multinomiale)
    - 'svm' : TF-IDF + LinearSVC calibré (Platt scaling)

    Returns:
        Dictionnaire {nom: Pipeline}.
    """
    # Pipeline Régression Logistique
    # Note : multi_class supprimé dans sklearn >= 1.7 (multinomial par défaut avec lbfgs)
    lr_pipe = Pipeline([
        ("tfidf", _make_tfidf()),
        ("clf", LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000,
            class_weight="balanced", solver="lbfgs",
        )),
    ])

    # Pipeline SVM calibré
    svm_pipe = Pipeline([
        ("tfidf", _make_tfidf()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, max_iter=2000, class_weight="balanced"),
            cv=3, method="sigmoid",
        )),
    ])

    return {"lr": lr_pipe, "svm": svm_pipe}


# ── Validation croisée ────────────────────────────────────────────────────────
def evaluer_cv(
    pipelines: dict[str, Pipeline],
    X: pd.Series,
    y: pd.Series,
    cv_splits: int = 5,
) -> dict[str, dict]:
    """
    Validation croisée stratifiée sur chaque pipeline.

    Args:
        pipelines : Dictionnaire {nom: Pipeline}.
        X         : Textes bruts.
        y         : Labels.
        cv_splits : Nombre de plis.

    Returns:
        Dictionnaire {nom: {metric: valeur_moyenne}}.
    """
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    resultats: dict[str, dict] = {}

    for nom, pipe in pipelines.items():
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=True,
        )
        resultats[nom] = {
            "accuracy_cv"       : float(scores["test_accuracy"].mean()),
            "f1_macro_cv"       : float(scores["test_f1_macro"].mean()),
            "f1_weighted_cv"    : float(scores["test_f1_weighted"].mean()),
            "accuracy_train_cv" : float(scores["train_accuracy"].mean()),
        }
        log.info(
            "[%s] CV %d-fold → acc=%.3f | macro-F1=%.3f | W-F1=%.3f",
            nom.upper(), cv_splits,
            resultats[nom]["accuracy_cv"],
            resultats[nom]["f1_macro_cv"],
            resultats[nom]["f1_weighted_cv"],
        )

    return resultats


# ── GridSearchCV (Régression Logistique) ─────────────────────────────────────
def grid_search_lr(X: pd.Series, y: pd.Series) -> Tuple[Pipeline, dict]:
    """
    GridSearchCV sur les hyper-paramètres TF-IDF + LR.

    Args:
        X : Textes bruts.
        y : Labels.

    Returns:
        (best_pipeline, best_params)
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True, min_df=1, max_df=0.95,
            stop_words=list(_STOP_WORDS_ACTIFS), analyzer="word",
        )),
        ("clf", LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000,
            class_weight="balanced", solver="lbfgs",
        )),
    ])

    param_grid = {
        "tfidf__ngram_range" : [(1, 1), (1, 2)],
        "tfidf__max_features": [None, 500, 1000],
        "clf__C"             : [0.01, 0.1, 1.0, 10.0],
    }

    gs = GridSearchCV(
        pipe, param_grid, cv=cv,
        scoring="f1_macro", n_jobs=-1, verbose=0,
    )
    gs.fit(X, y)

    log.info("GridSearch meilleurs params : %s", gs.best_params_)
    log.info("GridSearch meilleur macro-F1 : %.3f", gs.best_score_)

    return gs.best_estimator_, gs.best_params_


# ── Évaluation finale ────────────────────────────────────────────────────────
def evaluer_final(
    model: Pipeline,
    X: pd.Series,
    y: pd.Series,
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """
    Évaluation in-sample + matrice de confusion + analyse des erreurs
    + mots les plus discriminants.

    Args:
        model       : Pipeline entraîné sur tout le dataset.
        X           : Textes bruts.
        y           : Labels vrais.
        df          : DataFrame complet.
        figures_dir : Dossier de sortie des figures.
    """
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)

    # ── Rapport de classification ─────────────────────────────────────────────
    rapport = classification_report(y, y_pred, target_names=model.classes_, digits=3)
    print("\n" + "=" * 60)
    print("RAPPORT DE CLASSIFICATION (in-sample, n=24)")
    print("=" * 60)
    print(rapport)

    # ── Matrice de confusion ──────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred, labels=list(model.classes_))
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(
        "Matrice de confusion — Classifieur Sentiment\n(in-sample, n=24)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(figures_dir / "matrice_confusion.png", dpi=150)
    plt.close(fig)
    log.info("Figure sauvegardée : matrice_confusion.png")

    # ── Analyse des erreurs ───────────────────────────────────────────────────
    df_eval = df.copy()
    df_eval["prediction"] = y_pred
    df_eval["confiance"]  = y_proba.max(axis=1).round(3)
    erreurs = df_eval[df_eval["sentiment_ref"] != df_eval["prediction"]]

    print("\n" + "=" * 60)
    print(f"ANALYSE DES ERREURS : {len(erreurs)} erreur(s)")
    print("=" * 60)
    if erreurs.empty:
        print("  ✓ Aucune erreur in-sample (normal sur données d'entraînement)")
    else:
        for _, row in erreurs.iterrows():
            print(
                f"  [{row['id']}] VRAI={row['sentiment_ref']:8s} "
                f"→ PRÉDIT={row['prediction']:8s} "
                f"conf={row['confiance']:.2f} | {str(row['texte'])[:70]}"
            )

    # ── Mots discriminants (Régression Logistique uniquement) ─────────────────
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        print("\n" + "=" * 60)
        print("MOTS LES PLUS DISCRIMINANTS (Régression Logistique)")
        print("=" * 60)
        feature_names = model.named_steps["tfidf"].get_feature_names_out()
        for i, classe in enumerate(clf.classes_):
            top_idx  = np.argsort(clf.coef_[i])[::-1][:10]
            top_mots = [(feature_names[j], round(float(clf.coef_[i][j]), 3)) for j in top_idx]
            print(f"  [{classe.upper():8s}] →", ", ".join(f"{m}({v:+.2f})" for m, v in top_mots))


# ── Sauvegarde ────────────────────────────────────────────────────────────────
def sauvegarder_modele(model: Pipeline, model_dir: Path) -> None:
    """
    Sauvegarde le pipeline complet et le vectorizer séparé.

    Args:
        model     : Pipeline entraîné.
        model_dir : Dossier cible.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "model.pkl")
    log.info("Pipeline sauvegardé → %s", model_dir / "model.pkl")

    joblib.dump(model.named_steps["tfidf"], model_dir / "vectorizer.pkl")
    log.info("Vectorizer sauvegardé → %s", model_dir / "vectorizer.pkl")

    # Métadonnées lisibles
    meta = (
        f"Classifieur  : {type(model.named_steps['clf']).__name__}\n"
        f"Classes      : {list(model.classes_)}\n"
        f"Random state : {RANDOM_STATE}\n"
    )
    (model_dir / "model_meta.txt").write_text(meta, encoding="utf-8")
    log.info("Métadonnées sauvegardées → models/model_meta.txt")


# ── Point d'entrée ────────────────────────────────────────────────────────────
def main() -> None:
    """Pipeline d'entraînement complet et reproductible."""

    log.info("═" * 60)
    log.info("  CLASSIFIEUR DE SENTIMENT — ENTRAÎNEMENT")
    log.info("  Mini-Projet ML Appliqué 2026")
    log.info("═" * 60)

    # ── Étape 0 : Chargement & EDA ───────────────────────────────────────────
    df = charger_dataset(DATA_PATH)
    eda_et_graphiques(df, FIGURES_DIR)

    X: pd.Series = df["texte"]           # textes bruts (TF-IDF dans le pipeline)
    y: pd.Series = df["sentiment_ref"]

    print("\n" + "=" * 60)
    print("DISTRIBUTION DES CLASSES")
    print("=" * 60)
    print(y.value_counts().to_string())

    # ── Étape 1 : Comparaison LR vs SVM (validation croisée 5-fold) ──────────
    print("\n" + "=" * 60)
    print("VALIDATION CROISÉE STRATIFIÉE — 5-fold")
    print("=" * 60)
    pipelines = construire_pipelines()
    cv_res = evaluer_cv(pipelines, X, y, cv_splits=5)

    print(f"\n  {'Modèle':<8} {'Acc CV':>8} {'Macro-F1':>10} {'W-F1':>8}")
    print("  " + "-" * 36)
    for nom, res in cv_res.items():
        print(
            f"  {nom.upper():<8} {res['accuracy_cv']:>8.3f}"
            f" {res['f1_macro_cv']:>10.3f}"
            f" {res['f1_weighted_cv']:>8.3f}"
        )

    # ── Étape 2 : GridSearchCV sur Régression Logistique ─────────────────────
    print("\n" + "=" * 60)
    print("GRID SEARCH CV — Régression Logistique")
    print("=" * 60)
    best_pipeline, best_params = grid_search_lr(X, y)

    print(f"  Meilleurs params : {best_params}")

    # ── Étape 3 : Entraînement final sur 100% du dataset ─────────────────────
    log.info("Entraînement final sur %d exemples (100%% du dataset)…", len(X))
    best_pipeline.fit(X, y)

    # ── Étape 4 : Évaluation & figures ───────────────────────────────────────
    evaluer_final(best_pipeline, X, y, df, FIGURES_DIR)

    # ── Étape 5 : Sauvegarde ─────────────────────────────────────────────────
    sauvegarder_modele(best_pipeline, MODEL_DIR)

    log.info("═" * 60)
    log.info("  TERMINÉ  |  models/model.pkl  |  figures/")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
