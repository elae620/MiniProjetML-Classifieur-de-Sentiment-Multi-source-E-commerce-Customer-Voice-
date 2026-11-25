.PHONY: install train test run-api run-ui clean docker-build docker-run

# Installation des dépendances Python
install:
	pip install -r requirements.txt

# Entraînement complet et génération des artefacts
train:
	python train.py

# Exécution des tests automatisés avec couverture
test:
	pytest tests/test_api.py -v --cov=modules --cov=schemas --cov=services --cov=core

# Lancement de l'API REST FastAPI
run-api:
	uvicorn main:app --reload --host 127.0.0.1 --port 8000

# Lancement de l'application Streamlit
run-ui:
	streamlit run app_streamlit.py

# Nettoyage des caches Python et pytest
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Construction de l'image Docker
docker-build:
	docker build -t sentiment-multisource:1.0 .

# Exécution du conteneur Docker
docker-run:
	docker run -p 8000:8000 -p 8501:8501 sentiment-multisource:1.0
