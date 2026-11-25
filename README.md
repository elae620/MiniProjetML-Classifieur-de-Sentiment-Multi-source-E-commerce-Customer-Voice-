# 🛒 Classifieur de Sentiment Multi-source (E-commerce Customer Voice)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.163.1-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.67.0-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/pytest-40%20passed-brightgreen.svg)](tests/test_api.py)

Projet noté d'Examen — **Mini-Projet Machine Learning Appliqué 2026**  
Plateforme unifiée d'analyse de sentiment client multi-canal (Texte direct, Captures d'écran OCR, Messages vocaux ASR).

---

## 1. Présentation & Architecture Technique

Une plateforme e-commerce reçoit quotidiennement des avis clients sous trois formats hétérogènes :
1. **Texte** : formulaire web ou avis en ligne.
2. **Image** : capture d'écran d'un réseau social ou ticket scanné.
3. **Audio** : message vocal ou appel dicté au service client.

### Architecture en Cascade

Pour simplifier la complexité multimodale et éviter le surapprentissage sur de faibles volumes de données, les entrées hétérogènes sont d'abord normalisées en texte brut par des briques spécialisées, avant d'être soumises à un **classifieur supervisé unique**.

```
                   ┌────────────────────────────────────────┐
     Texte  ──────►│  POST /predict/text                    │──┐
                   │  (Pipeline TF-IDF + Régression Log.)   │  │
                   └────────────────────────────────────────┘  │
                   ┌────────────────────────────────────────┐  │
     Image  ──────►│  POST /predict/image                   │──┤
                   │  OCR (Pillow + Tesseract) ──► Modèle   │  │
                   └────────────────────────────────────────┘  ├──► Réponse JSON unifiée :
                   ┌────────────────────────────────────────┐  │    {
     Audio  ──────►│  POST /predict/audio                   │──┤      "source": "text|image|audio",
                   │  Transcription ASR (Whisper) ──► Modèle│  │      "texte_extrait": "...",
                   └────────────────────────────────────────┘  │      "sentiment_predit": "positif|negatif|neutre",
                   ┌────────────────────────────────────────┐  │      "confiance": 0.89
   Automatique ───►│  POST /predict (Détection auto du type)│──┘    }
                   └────────────────────────────────────────┘
```

---

## 2. Structure Complète du Projet

```
miniProjet_ML/
├── core/                               # Configuration centrale
│   ├── __init__.py
│   └── config.py                       # Constantes, seuils de taille, MIME autorisés
├── data/                               # Données sources et médias
│   └── dataset_sentiment_ecommerce/
│       ├── textes.csv                  # 24 avis d'entraînement étiquetés (8/8/8)
│       ├── images.csv                  # 9 images étiquetées avec texte attendu
│       ├── audios.csv                  # 9 audios étiquetés avec transcription
│       ├── images/                     # Fichiers I01.png à I09.png
│       └── audios/                     # Fichiers A01.mp3 à A09.mp3
├── docs/                               # Documentation pour l'oral
│   └── SOUTENANCE.md                   # Justifications, limites, 15 Q/R jury, script 5 min
├── figures/                            # Visualisations générées lors de l'entraînement
│   ├── distribution_classes.png
│   ├── longueur_textes.png
│   └── matrice_confusion.png
├── models/                             # Modèles sérialisés
│   ├── model.pkl                       # Pipeline scikit-learn complet (TF-IDF + LR)
│   ├── vectorizer.pkl                  # TfidfVectorizer sérialisé séparément
│   └── model_meta.txt                  # Métadonnées d'entraînement
├── modules/                            # Modules de traitement multimodal
│   ├── __init__.py
│   ├── classifier.py                   # Singleton du classifieur + fonction predict()
│   ├── ocr.py                          # Prétraitement Pillow + pytesseract (langue fra)
│   └── asr.py                          # Modèle Whisper singleton + résolution d'extensions
├── routers/                            # Contrôleurs API FastAPI
│   ├── __init__.py
│   ├── text.py                         # POST /predict/text
│   ├── image.py                        # POST /predict/image
│   ├── audio.py                        # POST /predict/audio
│   └── auto.py                         # POST /predict (détection automatique)
├── schemas/                            # Schémas de validation Pydantic v2
│   ├── __init__.py
│   └── prediction.py                   # TextRequest, UnifiedPredictionResponse, HealthResponse
├── services/                           # Logique métier et orchestration
│   ├── __init__.py
│   └── prediction_service.py           # Cascade OCR/ASR -> Modèle, gestion des erreurs
├── tests/                              # Suite de tests automatisés
│   ├── __init__.py
│   └── test_api.py                     # 40 tests unitaires et d'intégration (pytest)
├── app_streamlit.py                    # Interface web de test interactive (Streamlit)
├── Dockerfile                          # Conteneurisation de l'application
├── Makefile                            # Raccourcis de commandes courantes
├── main.py                             # Point d'entrée FastAPI (CORS, lifespan, /health)
├── pytest.ini                          # Configuration de l'environnement de test
├── requirements.txt                    # Dépendances Python figées
├── train.py                            # Script d'entraînement complet et reproductible
└── README.md                           # Documentation technique
```

---

## 3. Prérequis et Installation

### 3.1 Dépendances Système Externes

Le projet nécessite deux utilitaires système pour l'OCR et l'ASR :

#### 1. Tesseract OCR (Requis pour l'OCR d'images)
* **Windows** : Télécharger l'installateur UB-Mannheim : https://github.com/UB-Mannheim/tesseract/wiki  
  ⚠️ **Important** : Lors de l'installation, cochez le pack de langue **French (fra)** dans les composants optionnels.  
  Ajoutez `C:\Program Files\Tesseract-OCR` au `PATH` système si nécessaire.
* **Linux (Ubuntu/Debian)** :
  ```bash
  sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-fra
  ```
* **macOS** :
  ```bash
  brew install tesseract tesseract-lang
  ```

#### 2. FFmpeg (Requis pour la lecture audio MP3 par Whisper)
* **Windows** : `winget install Gyan.FFmpeg` ou télécharger sur https://ffmpeg.org
* **Linux** : `sudo apt-get install -y ffmpeg`
* **macOS** : `brew install ffmpeg`

### 3.2 Installation de l'Environnement Python

```bash
# 1. Cloner ou décompresser le projet et se placer à la racine
cd miniProjet_ML

# 2. Créer un environnement virtuel isolé
python -m venv venv

# 3. Activer l'environnement
# Sur Windows (PowerShell) :
.\venv\Scripts\Activate.ps1
# Sur Windows (CMD) :
.\venv\Scripts\activate.bat
# Sur Linux / macOS :
source venv/bin/activate

# 4. Installer les dépendances Python
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Guide d'Exécution

### 4.1 Ré-entraîner le Modèle ML
```bash
python train.py
```
* Génère l'EDA dans `figures/distribution_classes.png` et `figures/longueur_textes.png`.
* Exécute la validation croisée 5-fold et le GridSearchCV.
* Sauvegarde le pipeline dans `models/model.pkl` et `models/vectorizer.pkl`.
* Génère `figures/matrice_confusion.png`.

### 4.2 Lancer l'API FastAPI
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
* **API REST** disponible sur : http://127.0.0.1:8000
* **Documentation interactive Swagger UI** : http://127.0.0.1:8000/docs
* **Documentation ReDoc** : http://127.0.0.1:8000/redoc
* **Route de santé** : http://127.0.0.1:8000/health

### 4.3 Lancer l'Interface Web Streamlit
Dans un second terminal (avec l'environnement activé) :
```bash
streamlit run app_streamlit.py
```
* **Application Web** accessible sur : http://localhost:8501

### 4.4 Exécuter les Tests Automatisés
```bash
# Exécution simple des 40 tests
pytest tests/test_api.py -v

# Exécution avec rapport détaillé de couverture de code
pytest tests/test_api.py -v --cov=modules --cov=schemas --cov=services --cov=core
```

---

## 5. Exemples de Requêtes cURL

### 5.1 Vérification de Santé (Health Check)
```bash
curl -X GET "http://127.0.0.1:8000/health"
```
**Réponse attendue :**
```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### 5.2 Prédiction sur Texte Brut
```bash
curl -X POST "http://127.0.0.1:8000/predict/text" \
     -H "Content-Type: application/json" \
     -d '{"texte": "Produit conforme à la description, livraison super rapide en 24h, je recommande vivement !"}'
```
**Réponse :**
```json
{
  "source": "text",
  "texte_extrait": "Produit conforme à la description, livraison super rapide en 24h, je recommande vivement !",
  "sentiment_predit": "positif",
  "confiance": 0.3341
}
```

### 5.3 Prédiction depuis une Image (Capture OCR)
```bash
curl -X POST "http://127.0.0.1:8000/predict/image" \
     -F "file=@data/dataset_sentiment_ecommerce/images/I01.png"
```
**Réponse :**
```json
{
  "source": "image",
  "texte_extrait": "AVIS CLIENT VÉRIFIÉ Livraison express et produit parfait",
  "sentiment_predit": "positif",
  "confiance": 0.3341
}
```

### 5.4 Prédiction depuis un Fichier Audio (Message Vocal ASR)
```bash
curl -X POST "http://127.0.0.1:8000/predict/audio" \
     -F "file=@data/dataset_sentiment_ecommerce/audios/A01.mp3"
```
**Réponse :**
```json
{
  "source": "audio",
  "texte_extrait": "Bonjour, je suis vraiment ravi de ma commande, le produit est magnifique et livré très vite.",
  "sentiment_predit": "positif",
  "confiance": 0.3352
}
```

### 5.5 Prédiction avec Détection Automatique Polymorphe
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@data/dataset_sentiment_ecommerce/images/I02.png"
```

---

## 6. Résultats et Évaluation du Modèle

### Validation Croisée Stratifiée (5-fold)
| Modèle | Accuracy CV | Macro-F1 | Weighted-F1 | Décision |
|---|---|---|---|---|
| **Régression Logistique (L-BFGS)** | **0.380** | **0.324** | **0.317** | **Retenu (Meilleur compromis + probabilités)** |
| LinearSVC (Calibré Sigmoïde) | 0.370 | 0.298 | 0.351 | Écarté (calibration instable sur n=24) |

### Analyse des Termes les Plus Discriminants
* **Classe Négative** : `ne`, `pas`, `fuyez`, `remboursement`, `attente`, `incompatibilité`  
  *(Validation de la règle de conservation des négations lors du nettoyage NLP).*
* **Classe Neutre** : `conforme`, `relais`, `effectuée`, `notice`, `prévu`, `correct`
* **Classe Positive** : `qualité`, `ravi`, `satisfait`, `boutique`, `intuitive`, `fluide`

### Visualisations Disponibles
* Distribution des classes : `figures/distribution_classes.png`
* Longueur des avis : `figures/longueur_textes.png`
* Matrice de confusion in-sample : `figures/matrice_confusion.png`

---

## 7. Emplacements Prévus pour les Captures d'Écran

> *Pour votre rendu final et votre rapport de projet, insérez les captures d'écran réalisées dans votre environnement aux emplacements ci-dessous.*

### 7.1 Documentation Interactive Swagger UI (`/docs`)
<!-- INSERER_CAPTURE_SWAGGER_ICI : Capture montrant les routes /predict/text, /predict/image, /predict/audio, /predict et /health -->
_Emplacement réservé : Capture d'écran Swagger UI_

### 7.2 Interface Streamlit en Action
<!-- INSERER_CAPTURE_STREAMLIT_TEXTE_ICI : Capture de l'onglet Texte avec un avis positif -->
_Emplacement réservé : Capture d'écran de l'onglet 1 (Texte)_

<!-- INSERER_CAPTURE_STREAMLIT_IMAGE_ICI : Capture de l'onglet Image avec I01.png -->
_Emplacement réservé : Capture d'écran de l'onglet 2 (Image OCR)_

<!-- INSERER_CAPTURE_STREAMLIT_AUDIO_ICI : Capture de l'onglet Audio avec A01.mp3 -->
_Emplacement réservé : Capture d'écran de l'onglet 3 (Audio ASR)_

---

## 8. Déploiement Docker (Optionnel)

L'application est entièrement conteneurisable pour un déploiement agnostique de l'OS :

```bash
# Construire l'image Docker
docker build -t sentiment-ecommerce:1.0 .

# Lancer le conteneur
docker run -p 8000:8000 -p 8501:8501 sentiment-ecommerce:1.0
```
* Accès API : http://localhost:8000/docs
* Accès UI : http://localhost:8501
