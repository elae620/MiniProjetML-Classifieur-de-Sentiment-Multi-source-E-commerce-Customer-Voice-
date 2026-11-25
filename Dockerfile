# Dockerfile pour le Classifieur de Sentiment Multi-source
FROM python:3.11-slim

# Empêche Python d'écrire des fichiers .pyc et force l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Installation des dépendances système (Tesseract OCR avec langue française + ffmpeg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie de l'intégralité du code et des modèles
COPY . .

# Entraînement initial si les artefacts sont absents
RUN if [ ! -f models/model.pkl ]; then python train.py; fi

# Exposition des ports pour FastAPI (8000) et Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Commande par défaut : lancement de l'API FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
