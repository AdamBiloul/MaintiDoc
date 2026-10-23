# Exécuter MaintiDoc V1

Depuis `C:\Users\hp\Desktop\Maintidoc`, dans PowerShell.

## Installation

L'environnement et les modèles ont été installés sur cette machine. Pour une autre
installation, Python 3.13 et Ollama doivent être disponibles ; prévoir l'espace disque
pour les modèles et installer les dépendances :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt -c constraints-v1.txt
.\.venv\Scripts\python.exe scripts/download_bge.py
.\.venv\Scripts\python.exe scripts/start_ollama.py
.\.venv\Scripts\python.exe scripts/pull_llm.py
```

Les poids proviennent des dépôts officiels ; aucun compte d'API payant n'est requis.
La première instance Ollama existante sur le port 11434 n'est pas modifiée.
MaintiDoc utilise une instance dédiée sur le port 11435 et `.models/ollama/`.

## Préparer les identités et l'index

```powershell
.\.venv\Scripts\python.exe scripts/setup_demo.py
.\.venv\Scripts\python.exe -m maintidoc.ingest
```

Le compte de démonstration est créé une seule fois, sans écraser les identités
existantes. Pour ajouter un PDF, placez-le dans `data/raw/`, puis ajoutez une entrée
à `config/corpus.json` avec un identifiant unique, chemin, titre, version, famille,
`allowed_roles` et `authorized: true` uniquement si son utilisation est autorisée.

Le manifeste ne doit pas mélanger deux versions incompatibles d'un même équipement
sans les identifier explicitement. Le champ `version` reste une déclaration de
l'administrateur ; l'empreinte SHA-256 identifie le fichier réellement indexé.

Les PDF scannés nécessitent Tesseract avec les langues française/anglaise installées :

```powershell
.\.venv\Scripts\python.exe -m maintidoc.ingest --ocr
```

Sans `--ocr`, les pages vides sont recensées dans `data/index_v1.json`. Un document
entièrement sans texte est refusé. Le chemin OCR n'a pas été validé sur un corpus scanné.
Les tableaux complexes, schémas et images ne sont pas interprétés en V1.

## Démarrer

```powershell
.\.venv\Scripts\python.exe scripts/start_ollama.py
.\.venv\Scripts\python.exe scripts/start_demo.py
```

- Interface : http://127.0.0.1:8501
- Documentation API : http://127.0.0.1:8000/docs
- Santé de l'API : http://127.0.0.1:8000/health

Les questions peuvent prendre plus d'une minute sur CPU : le profil à mémoire limitée
recharge les modèles pour chaque question et les libère avant de passer au suivant.
Les journaux et PID sont dans `data/runtime/`, sans journalisation applicative des
questions, passages ou jetons. `/health` indique que l'API fonctionne, pas que chaque
réponse ou le modèle génératif sont disponibles.

Dans Streamlit, la session de démonstration locale est préremplie. Posez par exemple :
« Quels contrôles sont prévus avant la création d'un ticket ? ». Vérifiez les citations
en téléchargeant le PDF depuis la réponse et en allant à la page indiquée.
Une question sur une pompe ne peut pas être résolue tant que son manuel n'est pas indexé.

Pour un démarrage au premier plan dans deux terminaux, remplacez `start_demo.py` par :

```powershell
.\.venv\Scripts\python.exe -m uvicorn maintidoc.api:app --host 127.0.0.1 --port 8000 --workers 1
```

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/streamlit_app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
```

Arrêtez l'API avant réindexation ; Qdrant local verrouille sa base pour un seul processus.
Redémarrez l'API après modification du manifeste. Les scripts des anciennes séances
utilisent `data/qdrant/` et restent indépendants de la V1 (`data/qdrant_v1/`).

## Vérifier

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/evaluate_v1.py
```

Le premier contrôle utilise des fixtures synthétiques isolées, sans modèle.
Le second exige les services démarrés et utilise BGE-M3, Qwen et le PDF réellement indexé.
Il enregistre les réponses, temps observés et résultats dans `data/evaluation/v1_live.json`.
Ces résultats de démonstration ne remplacent pas une évaluation sur des manuels métier.

## Paramètres

- `MAINTIDOC_THRESHOLD` : seuil de recherche expérimental (défaut 0,40).
- `MAINTIDOC_OLLAMA_URL` : serveur Ollama (défaut `http://127.0.0.1:11435`).
- `MAINTIDOC_LLM` : modèle génératif (défaut `qwen2.5:1.5b`).
- `MAINTIDOC_API_URL` : adresse API utilisée par Streamlit.

Ne publiez pas la démonstration sur Internet avec ses identités locales et son jeton prérempli.
Le client pose des questions textuelles ; photographies et fine-tuning appartiennent à V2.
