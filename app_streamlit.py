"""
app_streamlit.py — Interface de test Streamlit pour l'API de classification de sentiment.

Lancement :
    1. Démarrer l'API :  uvicorn main:app --reload --port 8000
    2. Démarrer l'UI  :  streamlit run app_streamlit.py

Fonctionnalités :
    - 4 onglets : Texte | Image | Audio | Auto
    - Appel de l'API FastAPI en arrière-plan (localhost:8000)
    - Affichage du texte extrait (OCR/ASR), sentiment (couleur + emoji) et barre de confiance
    - Gestion des erreurs API avec messages explicites
"""

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE_URL = "http://localhost:8000"

# Mapping sentiment → emoji + couleur CSS
SENTIMENT_CONFIG = {
    "positif": {"emoji": "😊", "couleur": "#2ecc71", "label": "Positif"},
    "negatif": {"emoji": "😠", "couleur": "#e74c3c", "label": "Négatif"},
    "neutre":  {"emoji": "😐", "couleur": "#3498db", "label": "Neutre"},
}


# ── Fonctions utilitaires ─────────────────────────────────────────────────────
def verifier_api() -> bool:
    """Vérifie que l'API FastAPI est accessible."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def afficher_resultat(data: dict) -> None:
    """
    Affiche le résultat de prédiction de manière visuelle :
    - Texte extrait dans un encadré
    - Sentiment avec emoji et couleur
    - Barre de confiance avec pourcentage
    """
    sentiment = data.get("sentiment_predit", "neutre")
    confiance = data.get("confiance", 0.0)
    texte_extrait = data.get("texte_extrait", "")
    source = data.get("source", "?")
    config = SENTIMENT_CONFIG.get(sentiment, SENTIMENT_CONFIG["neutre"])

    # Encadré texte extrait
    st.markdown("#### 📄 Texte analysé")
    st.info(texte_extrait if texte_extrait else "_Aucun texte extrait_")

    # Sentiment avec emoji et couleur
    st.markdown("#### 🎯 Sentiment prédit")
    st.markdown(
        f"""
        <div style="
            background-color: {config['couleur']}20;
            border-left: 5px solid {config['couleur']};
            padding: 15px 20px;
            border-radius: 8px;
            margin: 10px 0;
        ">
            <span style="font-size: 2.5em;">{config['emoji']}</span>
            <span style="font-size: 1.8em; font-weight: bold; color: {config['couleur']}; margin-left: 10px;">
                {config['label']}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Barre de confiance
    st.markdown("#### 📊 Indice de confiance")
    pct = round(confiance * 100, 1)
    st.progress(confiance)
    st.markdown(
        f"<p style='text-align:center; font-size:1.2em; font-weight:bold;'>"
        f"{pct}%</p>",
        unsafe_allow_html=True,
    )

    # Source détectée
    st.caption(f"Source détectée : **{source}**")


def afficher_erreur(response) -> None:
    """Affiche un message d'erreur API clair."""
    try:
        detail = response.json()
        msg = detail.get("detail", str(detail))
    except Exception:
        msg = response.text or f"Erreur HTTP {response.status_code}"
    st.error(f"❌ Erreur API ({response.status_code}) : {msg}")


# ── Page principale ───────────────────────────────────────────────────────────
def main():
    # Configuration de la page
    st.set_page_config(
        page_title="🛒 Classifieur Sentiment E-commerce",
        page_icon="🛒",
        layout="centered",
    )

    st.title("🛒 Classifieur de Sentiment Multi-source")
    st.markdown(
        "Analysez le sentiment d'avis clients depuis **du texte**, "
        "**une image** (OCR) ou **un fichier audio** (ASR)."
    )

    # Vérification de la connexion API
    api_ok = verifier_api()
    if api_ok:
        st.sidebar.success("✅ API connectée")
    else:
        st.sidebar.error(
            "❌ API non accessible\n\n"
            "Lancez l'API avec :\n"
            "```bash\n"
            "uvicorn main:app --reload --port 8000\n"
            "```"
        )

    # Sidebar : infos techniques
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Documentation API")
    st.sidebar.markdown(
        f"- [Swagger UI]({API_BASE_URL}/docs)\n"
        f"- [ReDoc]({API_BASE_URL}/redoc)"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Mini-Projet ML Appliqué 2026**\n\n"
        "Architecture en cascade :\n"
        "- Image → OCR → Texte → Modèle\n"
        "- Audio → ASR → Texte → Modèle\n"
        "- Texte → Modèle"
    )

    # ── Onglets ───────────────────────────────────────────────────────────────
    tab_texte, tab_image, tab_audio, tab_auto = st.tabs(
        ["📝 Texte", "🖼️ Image", "🎙️ Audio", "🔄 Auto"]
    )

    # ── Onglet 1 : Texte ─────────────────────────────────────────────────────
    with tab_texte:
        st.header("📝 Analyse de texte")
        st.markdown("Saisissez ou collez un avis client pour analyser son sentiment.")

        texte_input = st.text_area(
            "Avis client :",
            placeholder="Ex : Produit conforme à la description, livraison rapide et soignée !",
            height=120,
            key="texte_input",
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            btn_texte = st.button("🔍 Analyser", key="btn_texte", type="primary")

        if btn_texte:
            if not texte_input or not texte_input.strip():
                st.warning("⚠️ Veuillez saisir un texte avant de lancer l'analyse.")
            elif not api_ok:
                st.error("❌ L'API n'est pas accessible. Veuillez la démarrer.")
            else:
                with st.spinner("Analyse en cours..."):
                    try:
                        r = requests.post(
                            f"{API_BASE_URL}/predict/text",
                            json={"texte": texte_input.strip()},
                            timeout=30,
                        )
                        if r.status_code == 200:
                            afficher_resultat(r.json())
                        else:
                            afficher_erreur(r)
                    except requests.ConnectionError:
                        st.error("❌ Connexion à l'API impossible.")
                    except requests.Timeout:
                        st.error("⏰ Délai d'attente dépassé.")

    # ── Onglet 2 : Image ─────────────────────────────────────────────────────
    with tab_image:
        st.header("🖼️ Analyse d'image (OCR)")
        st.markdown(
            "Importez une image contenant un avis client "
            "(capture d'écran, ticket scanné, post réseau social)."
        )

        image_file = st.file_uploader(
            "Choisir une image :",
            type=["png", "jpg", "jpeg", "webp", "bmp"],
            key="image_upload",
        )

        if image_file:
            st.image(image_file, caption="Image importée", use_container_width=True)

        col1, col2 = st.columns([1, 4])
        with col1:
            btn_image = st.button("🔍 Analyser", key="btn_image", type="primary")

        if btn_image:
            if not image_file:
                st.warning("⚠️ Veuillez importer une image.")
            elif not api_ok:
                st.error("❌ L'API n'est pas accessible.")
            else:
                with st.spinner("Extraction OCR et analyse en cours..."):
                    try:
                        files = {
                            "file": (
                                image_file.name,
                                image_file.getvalue(),
                                image_file.type or "image/png",
                            )
                        }
                        r = requests.post(
                            f"{API_BASE_URL}/predict/image",
                            files=files,
                            timeout=60,
                        )
                        if r.status_code == 200:
                            afficher_resultat(r.json())
                        else:
                            afficher_erreur(r)
                    except requests.ConnectionError:
                        st.error("❌ Connexion à l'API impossible.")
                    except requests.Timeout:
                        st.error("⏰ Délai d'attente dépassé (OCR peut être long).")

    # ── Onglet 3 : Audio ─────────────────────────────────────────────────────
    with tab_audio:
        st.header("🎙️ Analyse audio (ASR)")
        st.markdown(
            "Importez un fichier audio contenant un avis vocal "
            "(message téléphone, enregistrement service client)."
        )

        audio_file = st.file_uploader(
            "Choisir un fichier audio :",
            type=["wav", "mp3", "ogg", "m4a", "flac", "webm"],
            key="audio_upload",
        )

        if audio_file:
            st.audio(audio_file, format=f"audio/{audio_file.name.split('.')[-1]}")

        col1, col2 = st.columns([1, 4])
        with col1:
            btn_audio = st.button("🔍 Analyser", key="btn_audio", type="primary")

        if btn_audio:
            if not audio_file:
                st.warning("⚠️ Veuillez importer un fichier audio.")
            elif not api_ok:
                st.error("❌ L'API n'est pas accessible.")
            else:
                with st.spinner("Transcription ASR et analyse en cours (peut prendre 10-30s)..."):
                    try:
                        files = {
                            "file": (
                                audio_file.name,
                                audio_file.getvalue(),
                                audio_file.type or "audio/mpeg",
                            )
                        }
                        r = requests.post(
                            f"{API_BASE_URL}/predict/audio",
                            files=files,
                            timeout=120,
                        )
                        if r.status_code == 200:
                            afficher_resultat(r.json())
                        else:
                            afficher_erreur(r)
                    except requests.ConnectionError:
                        st.error("❌ Connexion à l'API impossible.")
                    except requests.Timeout:
                        st.error("⏰ Délai d'attente dépassé (Whisper peut prendre du temps).")

    # ── Onglet 4 : Auto ──────────────────────────────────────────────────────
    with tab_auto:
        st.header("🔄 Détection automatique")
        st.markdown(
            "Importez n'importe quel fichier (image ou audio) et l'API "
            "détectera automatiquement le type pour appliquer le bon pipeline."
        )

        auto_file = st.file_uploader(
            "Choisir un fichier (image ou audio) :",
            type=["png", "jpg", "jpeg", "webp", "bmp", "wav", "mp3", "ogg", "m4a", "flac"],
            key="auto_upload",
        )

        if auto_file:
            file_ext = auto_file.name.split(".")[-1].lower()
            if file_ext in {"png", "jpg", "jpeg", "webp", "bmp"}:
                st.image(auto_file, caption="Fichier détecté : Image", use_container_width=True)
            else:
                st.audio(auto_file, format=f"audio/{file_ext}")
                st.caption("Fichier détecté : Audio")

        col1, col2 = st.columns([1, 4])
        with col1:
            btn_auto = st.button("🔍 Analyser", key="btn_auto", type="primary")

        if btn_auto:
            if not auto_file:
                st.warning("⚠️ Veuillez importer un fichier.")
            elif not api_ok:
                st.error("❌ L'API n'est pas accessible.")
            else:
                with st.spinner("Détection et analyse en cours..."):
                    try:
                        files = {
                            "file": (
                                auto_file.name,
                                auto_file.getvalue(),
                                auto_file.type or "application/octet-stream",
                            )
                        }
                        r = requests.post(
                            f"{API_BASE_URL}/predict",
                            files=files,
                            timeout=120,
                        )
                        if r.status_code == 200:
                            afficher_resultat(r.json())
                        else:
                            afficher_erreur(r)
                    except requests.ConnectionError:
                        st.error("❌ Connexion à l'API impossible.")
                    except requests.Timeout:
                        st.error("⏰ Délai d'attente dépassé.")


if __name__ == "__main__":
    main()
