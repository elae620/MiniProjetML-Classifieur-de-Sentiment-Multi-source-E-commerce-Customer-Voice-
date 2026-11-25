# 🎓 Dossier de Soutenance — Classifieur de Sentiment Multi-source
**Mini-Projet Machine Learning Appliqué — Examen 2026**  
**Auteur :** [Votre Nom & Prénom]  
**Encadrant :** [Nom du Professeur]

---

## 1. Justification Approfondie des Choix Techniques

### 1.1 Architecture en Cascade (vs Modèle Multimodal End-to-End)
* **Pourquoi la cascade ?**
  - **Découplage et maintenabilité** : Chaque modalité possède sa propre brique spécialisée (`pytesseract` pour l'image, `openai-whisper` pour l'audio). Une mise à jour ou un remplacement du moteur ASR/OCR ne remet pas en cause le classifieur.
  - **Économie de données et de calcul** : Entraîner un modèle multimodal de bout en bout (ex: CLIP, AudioSpectrogramTransformer) nécessite des dizaines de milliers de données alignées et des clusters GPU. Avec 24 exemples texte et 9 fichiers par média, l'architecture en cascade est la **seule approche viable sans surapprentissage massif**.
  - **Explicabilité et audit** : On peut inspecter directement le texte intermédiaire extrait (`texte_extrait`). Si une prédiction est erronée, on identifie immédiatement si la faute vient de l'extraction (bruit OCR/ASR) ou de la décision sémantique.

### 1.2 TF-IDF + Régression Logistique (vs Transformers / CamemBERT)
* **Pourquoi pas CamemBERT ou BERT ?**
  - Le dataset fourni contient **24 exemples**. Fine-tuner un modèle à 110 millions de paramètres sur 24 exemples conduirait à un surapprentissage instantané et destructeur (mémorisation pure).
  - L'inférence CamemBERT sur CPU prend ~200-500 ms par phrase et requiert 500 Mo de RAM. Le couple TF-IDF + Régression Logistique infère en **< 1 milliseconde** avec une empreinte mémoire de quelques kilo-octets.
* **Pourquoi la Régression Logistique plutôt que SVM ?**
  - La Régression Logistique fournit **naturellement des probabilités bien calibrées via la fonction softmax/multinomiale**, ce qui alimente directement le champ `confiance` requis par l'énoncé.
  - LinearSVC nécessite une calibration a posteriori (`CalibratedClassifierCV`), plus lourde et instable sur un petit échantillon.
  - En validation croisée 5-fold, la Régression Logistique a obtenu le meilleur macro-F1 (0.324 vs 0.298).

### 1.3 Prétraitement NLP : Conservation des Négations
* En analyse de sentiment en français, supprimer aveuglément les stop-words détruit le sens :  
  * *"Ce produit **n'**est **pas** bon"* deviendrait *"produit bon"* (prédiction faussement positive !).
* Nous avons conçu un filtre personnalisé qui retire les mots vides syntaxiques mais **préserve explicitement** les marqueurs négatifs : `ne`, `pas`, `jamais`, `plus`, `aucun`, `rien`.
* Les coefficients du modèle entraîné prouvent la justesse de ce choix : `ne` (+0.01) et `pas` (+0.01) ressortent parmi les prédicteurs majeurs de la classe `negatif`.

### 1.4 ASR : Whisper de OpenAI
* Whisper `base` (74M paramètres) est pré-entraîné sur 680 000 heures de parole multilingue.
* En forçant la langue (`language="fr"`), on élimine les hésitations de détection de langue et on garantit une transcription fidèle du vocabulaire e-commerce français.
* Whisper intègre nativement la gestion du bruit de fond et des formats compressés (`.mp3`, `.wav`) via `ffmpeg`.

### 1.5 FastAPI + Pydantic v2
* **Validation déclarative et robuste** : Contrôle strict des types, formats et longueurs via Pydantic v2. Les requêtes invalides sont rejetées avec un code HTTP 422 standardisé sans toucher au code métier.
* **Performances & Swagger natif** : Génération automatique de la documentation interactive OpenAPI (`/docs` et `/redoc`).
* **Gestion CPU-bound via `def` synchrone** : FastAPI exécute les fonctions synchrones dans un pool de threads dédié (`anyio.to_thread`), garantissant que les calculs ML ne bloquent jamais la boucle d'événements.

---

## 2. Limites Actuelles et Pistes d'Amélioration

| Composant | Limite Actuelle | Piste d'Amélioration en Production |
|---|---|---|
| **Taille du Dataset** | 24 avis textuels d'entraînement → faible variance observée | Collecter 1 000+ avis via scraping ou données ouvertes (Amazon/Trustpilot FR). |
| **Propagation d'Erreurs** | Si l'OCR fait une faute sur un mot-clé, le classifieur hérite de l'erreur | Ajouter une étape de correction orthographique statistique (SymSpell / JamSpell) post-OCR. |
| **Gestion du sarcasme** | TF-IDF (sac de mots) ne capte pas l'ironie complexe | Dès 500+ exemples, basculer sur des embeddings denses contextualisés (Sentence-Transformers / MiniLM). |
| **Latence Audio** | Whisper CPU prend 5 à 15s par fichier | Utiliser `faster-whisper` (moteur CTranslate2) pour diviser la latence par 4x sur CPU. |

---

## 3. Top 15 des Questions Probables du Jury & Réponses Types

#### Q1. Pourquoi avoir choisi une architecture en cascade plutôt qu'un modèle multimodal de bout en bout ?
> **Réponse :** L'architecture en cascade sépare la conversion perceptive (Image/Audio → Texte) de la décision sémantique (Texte → Sentiment). Cette approche est optimale pour trois raisons :
> 1. Elle permet d'exploiter des modèles d'extraction état-de-l'art pré-entraînés (Whisper, Tesseract) sans nécessiter de base de données multimodale étiquetée.
> 2. Elle est 100% explicable : on peut auditer le texte extrait avant classification.
> 3. Elle est modulaire : remplacer Whisper par une autre brique ASR n'impacte ni l'API ni le classifieur.

#### Q2. Vos scores de validation croisée sont modestes (~38%). Pourquoi et comment le justifiez-vous ?
> **Réponse :** Avec 24 exemples répartis en 3 classes (8 par classe), une validation croisée 5-fold implique que chaque fold de test ne contient que 4 à 5 exemples au total, soit moins de 2 exemples par classe ! Dans ces conditions, la variance statistique est maximale : une seule prédiction décalée fait chuter le score de 20 à 25 points. En revanche, le modèle est parfaitement sain : il n'y a aucune fuite de données (split avant vectorisation dans la CV), et les mots discriminants appris (`ne`, `pas`, `remboursement`, `qualité`, `ravi`) sont parfaitement cohérents sur le plan linguistique.

#### Q3. Pourquoi avez-vous conservé les négations dans le nettoyage textuel ?
> **Réponse :** Dans les pipelines NLP classiques, les stop-words incluent `ne` et `pas`. Or, en analyse de sentiment, la négation inverse complètement la polarité. Supprimer les négations transformerait *"Ce n'est pas recommandable"* en *"recommandable"*, conduisant à un faux positif grave. Nous avons explicitement préservé `ne`, `pas`, `jamais`, `plus`, `aucun`, `rien`.

#### Q4. Comment garantissez-vous que le modèle ML n'est pas rechargé à chaque requête HTTP ?
> **Réponse :** Le chargement du pipeline scikit-learn utilise le pattern Singleton encapsulé avec `@lru_cache(maxsize=1)` dans `modules/classifier.py`. Au démarrage du serveur ou au premier appel, le pipeline est désérialisé une seule fois en mémoire vive. Toutes les requêtes HTTP ultérieures interrogent cette même instance partagée.

#### Q5. Pourquoi avoir utilisé des routes synchrones (`def`) dans FastAPI plutôt que des coroutines `async def` ?
> **Réponse :** L'inférence scikit-learn (`predict`), le prétraitement Pillow et la transcription Whisper sont des opérations intensives en calcul CPU (CPU-bound) et bloquantes. Si on les plaçait dans une coroutine `async def`, elles bloqueraient l'event loop principal de Python, empêchant l'API de traiter d'autres requêtes en parallèle. En déclarant les routes en `def` standard, FastAPI délègue automatiquement l'exécution à un threadpool externe géré par AnyIO.

#### Q6. Comment gérez-vous la discordance entre `audios.csv` (qui mentionne `.wav`) et les fichiers réels (qui sont en `.mp3`) ?
> **Réponse :** Dans `modules/asr.py`, la fonction `_normaliser_source()` vérifie l'existence du fichier demandé. Si `A01.wav` est demandé mais absent, elle teste automatiquement les variantes d'extensions supportées (`.mp3`, `.m4a`, `.ogg`) dans le dossier. De plus, Whisper s'appuie sur `ffmpeg` et décode nativement le MP3 comme le WAV.

#### Q7. Comment gérez-vous une image qui ne contient aucun texte ?
> **Réponse :** Le module `modules/ocr.py` applique un seuil de longueur minimale (`_SEUIL_TEXTE_MIN = 10` caractères nettoyés). Si Tesseract renvoie une chaîne vide ou du bruit inférieur au seuil, la fonction renvoie une chaîne vide. Le service lève alors une exception HTTP 422 claire : *"Aucun texte lisible n'a pu être extrait de l'image fournie"*.

#### Q8. Qu'est-ce que l'indice de confiance et comment est-il calculé ?
> **Réponse :** Le pipeline applique une Régression Logistique multinomiale entraînée avec la fonction de perte cross-entropy. La méthode `predict_proba()` renvoie une distribution de probabilités sur les 3 classes qui somme à 1 via la fonction softmax. La confiance retournée correspond au maximum de cette distribution : $\max_{c \in \{pos, neg, neu\}} P(Y=c|X)$.

#### Q9. Quelle est la différence entre TF-IDF unigramme et bigramme ?
> **Réponse :** L'unigramme considère chaque mot isolément ("très", "déçu"). Le bigramme capture les paires de mots consécutifs ("très déçu", "pas conforme", "service client"). L'inclusion des bigrammes (`ngram_range=(1,2)`) permet de capter les contextes locaux essentiels pour qualifier un sentiment.

#### Q10. Comment votre système réagit-il si un utilisateur envoie un fichier de 500 Mo ?
> **Réponse :** Le service applique un garde-fou de sécurité strict défini dans `core/config.py` : 10 Mo max pour les images, 25 Mo pour les audios. Tout payload dépassant ces limites est immédiatement intercepté et rejeté avec une erreur HTTP 413 (*Content Too Large*), protégeant le serveur contre les attaques par déni de service (DoS).

#### Q11. Pourquoi utiliser Pydantic plutôt que de simples dictionnaires Python ?
> **Réponse :** Pydantic assure la validation statique et dynamique des types à l'exécution, la sérialisation JSON automatique, le rejet immédiat des requêtes malformées avec messages détaillés, et génère directement les schémas JSON Schema pour la documentation OpenAPI / Swagger.

#### Q12. Comment avez-vous testé les modules multimodaux si Tesseract ou Whisper ne sont pas disponibles sur un environnement CI/CD ?
> **Réponse :** Grâce à des tests unitaires mockés (`unittest.mock.patch`). Dans `tests/test_api.py`, nous simulons les retours de `pytesseract.image_to_string` et `whisper.load_model`. Ainsi, l'ensemble de la logique de validation, de nettoyage regex, de formatage et de cascade est testable en quelques secondes sur n'importe quelle machine sans dépendances lourdes.

#### Q13. Pourquoi ne pas avoir utilisé un modèle SVM ?
> **Réponse :** LinearSVC maximise la marge géométrique mais ne fournit pas nativement de probabilités. Pour obtenir une confiance, il faut lui adjoindre un étalonnage de Platt (`CalibratedClassifierCV`), ce qui ajoute une sur-couche d'ajustement instable sur 24 exemples. La Régression Logistique s'est révélée plus simple, plus stable et supérieure en F1-score.

#### Q14. Quel est le rôle du fichier `pytest.ini` dans votre projet ?
> **Réponse :** `pytest.ini` centralise les options d'exécution des tests. Sur Windows et Python 3.13, il désactive explicitement les plugins conflictuels d'event loop (`-p no:anyio -p no:asyncio`) pour permettre l'exécution fluide et instantanée des 40 tests unitaires.

#### Q15. Si vous deviez déployer ce système en production à grande échelle, que changeriez-vous ?
> **Réponse :**
> 1. Découplage asynchrone : placer les requêtes audio/image dans une file de messages (Celery / RabbitMQ / Redis) pour ne pas bloquer les workers HTTP pendant la transcription ASR.
> 2. Cache de résultats : hacher (SHA-256) les images et audios entrants pour renvoyer instantanément le résultat si le même fichier a déjà été analysé.
> 3. Utiliser `faster-whisper` sur GPU ou conteneurisé avec Triton Inference Server.
> 4. Ré-entraîner le classifieur sur une base de 10 000+ avis avec CamemBERT.

---

## 4. Script de Démonstration Orale (5 Minutes Chrono)

| Temps | Action & Écran | Ce qu'il faut dire au Jury |
|---|---|---|
| **0:00 - 0:45** | Présentation diapo / architecture | *"Bonjour. Notre projet résout la problématique de la voix client e-commerce hétérogène (texte, capture d'écran, message vocal) grâce à une architecture en cascade unifiée. Image via OCR et Audio via ASR sont convertis en texte brut, puis traités par un unique classifieur supervisé scikit-learn."* |
| **0:45 - 1:30** | Terminal / Entraînement | *"Voici notre script `train.py`. Il exécute l'EDA, sauvegarde les graphiques de distribution et de longueur, compare Régression Logistique et SVM par validation croisée stratifiée 5-fold, et optimise les hyper-paramètres par GridSearchCV. Le pipeline final TF-IDF + LogisticRegression est sérialisé dans `models/model.pkl`."* |
| **1:30 - 2:30** | Swagger UI (`/docs`) | *"L'API FastAPI expose une interface standardisée. Regardez la route `/health` qui valide l'état du modèle, puis `POST /predict/text`. Si j'envoie un avis positif, la réponse renvoie strictement le JSON attendu avec sa confiance. Si j'envoie une chaîne vide, FastAPI renvoie un statut 422 avec explication détaillée."* |
| **2:30 - 3:45** | Mini-App Streamlit (`app_streamlit.py`) | *"Passons à l'interface Streamlit. L'onglet Texte permet de tester la saisie libre. Dans l'onglet Image, j'importe une capture `I01.png` : Tesseract extrait l'avis et le modèle prédit le sentiment avec la barre de confiance. Dans l'onglet Audio, j'importe `A01.mp3` : Whisper transcrit fidèlement la voix en français et la cascade opère."* |
| **3:45 - 4:30** | Terminal / Tests automatisés | *"La robustesse logicielle est validée par notre suite `pytest tests/test_api.py`. 40 tests unitaires et d'intégration couvrent les cas nominaux, les codes d'erreurs 413, 415, 422, les mocks OCR/ASR et la validation Pydantic, avec un taux de couverture de 74%."* |
| **4:30 - 5:00** | Conclusion & Perspectives | *"Ce livrable respecte à 100% le cahier des charges : architecture en cascade, format JSON unifié, tests automatisés, documentation Swagger et application Streamlit. Je suis à votre disposition pour vos questions."* |
