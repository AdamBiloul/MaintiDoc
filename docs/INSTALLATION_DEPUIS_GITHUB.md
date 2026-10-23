# Installer depuis GitHub

Le dépôt contient le code, les tests et les guides. Les PDF, modèles, index, données
d'adaptation, rapports locaux, identités et jetons ne sont pas publiés. Les résultats
historiques de démonstration sont décrits dans les bilans ; leurs fichiers locaux ne
sont pas présents dans un nouveau clone.

Depuis PowerShell, dans le dossier cloné :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
Copy-Item config/corpus.example.json config/corpus.json
```

Installer PyTorch adapté à la machine si nécessaire. Les versions observées sur la
machine d'origine sont dans `constraints-v2.txt` et `environment-versions.json` ;
elles ne constituent pas une garantie de compatibilité sur tout autre système.

Ajouter les PDF autorisés à `data/raw`, remplir le manifeste avec leurs références,
versions et droits, puis passer `authorized` à `true` seulement après vérification.
La feuille de route utilisée dans la démonstration n'est pas distribuée avec ce dépôt.

```powershell
.\.venv\Scripts\python.exe scripts/download_bge.py
.\.venv\Scripts\python.exe scripts/download_vision.py
.\.venv\Scripts\python.exe scripts/download_clip.py
.\.venv\Scripts\python.exe scripts/download_reranker.py
.\.venv\Scripts\python.exe scripts/setup_demo.py
.\.venv\Scripts\python.exe -m maintidoc.ingest
.\.venv\Scripts\python.exe -m maintidoc.figures
```

Installer Ollama, puis préparer le générateur local selon le guide V1 :

```powershell
.\.venv\Scripts\python.exe scripts/start_ollama.py
.\.venv\Scripts\python.exe scripts/pull_llm.py
.\.venv\Scripts\python.exe scripts/start_demo.py
```

Ouvrir http://127.0.0.1:8501. La session de démonstration est locale ; son jeton n'est
jamais à publier. Pour les tests indépendants des modèles et des PDF privés :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Pour reproduire l'adaptation fictive et les métriques, suivre `ETAPE_2_BILAN.md`.
`validate_stage2.py` nécessite également les rapports des recettes exécutées sur
votre propre corpus ; il n'est pas un test d'installation d'un clone vierge.
