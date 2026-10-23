# Étape 2 — photographie et explorateur de figures

Le prototype comprend désormais le reranker et l'adaptation CPU. Voir le
[bilan de clôture technique](ETAPE_2_BILAN.md) pour les commandes, les résultats
mesurés et les données métier qui restent nécessaires à la validation finale.

## Ce qui fonctionne

L'interface accepte une question et une photographie JPEG/PNG. L'endpoint authentifié
`POST /analyze-image` reçoit la question, l'image en base64 et, éventuellement, une
référence d'équipement. SmolVLM-256M produit une observation locale **non vérifiée en
anglais**. Cette observation ne constitue pas une preuve documentaire ni un diagnostic.

Sans référence exacte ou manuel associé accessible, la réponse demande une clarification.
Avec un manuel associé, la recherche reste filtrée par identité ET par équipement,
puis réutilise les citations vérifiées de V1. Le modèle ne devine pas la référence
depuis l'image et aucun outil d'action n'est exposé.

Un second incrément ajoute l'extraction des régions raster/vectorielles des PDF,
leur indexation CLIP et un explorateur qui fusionne les classements du texte, des
figures recherchées en français et d'une image facultative. Les pages, versions,
extraits textuels et aperçus autorisés sont affichés ensemble.
Cet explorateur présente des références candidates : il ne génère pas une
interprétation technique du schéma et ne remplace pas la réponse documentaire de V1.

Le modèle choisi est un prototype léger pour cette machine CPU : environ 513 Mo de
poids au lieu d'un VLM de plusieurs milliards de paramètres. La révision téléchargée
est consignée dans `data/evaluation/vision_model.json`. Ce choix ne garantit pas les
capacités du modèle Qwen2.5-VL-7B donné en exemple dans la feuille de route.

## Utiliser

Les modèles et services sont déjà préparés sur cette machine. Ouvrir
http://127.0.0.1:8501, saisir la question, ajouter une photo et, si connue, la référence.

Pour installer le premier incrément sur une autre machine :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe scripts/download_vision.py
```

Redémarrer l'API après une modification de son code. Le téléchargement est la seule
étape réseau du VLM : l'inférence utilise les fichiers locaux, sans code distant autorisé.
Le modèle est chargé dans un processus éphémère ; il se termine avant les embeddings
et le générateur textuel afin de limiter le pic mémoire.

## Associer les manuels aux équipements

Lorsqu'un vrai manuel autorisé est disponible, ajouter à son entrée du manifeste :

```json
"equipment_ids": ["REFERENCE-EXACTE-DU-MODELE"]
```

Puis arrêter l'API, réindexer avec `python -m maintidoc.ingest` et redémarrer.
Ne pas associer la feuille de route à une pompe réelle : elle n'est pas un manuel.
Les identifiants sont comparés sans tenir compte de la casse, sans rapprochement flou.
L'association est déclarée par l'administrateur, pas confirmée par la photographie.

## Contrôles et tests

- Corps HTTP limité en lecture à 8 Mo ; image décodée limitée à 5 Mo et 12 mégapixels.
- Formats réellement décodés : JPEG/PNG, sans recours à une URL externe.
- Correction d'orientation, conversion RGB, réduction à 1 024 pixels de côté maximum.
- Réencodage sans EXIF ni métadonnées textuelles.
- Aucune photographie enregistrée par l'API ; le navigateur conserve seulement son upload en session.
- Observation et légendes d'image traitées comme des données non fiables.
- Conservation des contrôles d'accès aux passages et aux PDF.

Les tests V1 et V2 passent ensemble ; le bilan V2 donne les vérifications finales.
Les tests du module photographie couvrent les
formats, tailles et métadonnées, la clarification, le refus d'accès au manuel d'un
autre rôle, le filtrage documentaire lors d'une question liée à un équipement et le
rendu distinct de l'observation visuelle et de la clarification dans Streamlit.

Le VLM a également été exécuté sur un **dessin géométrique synthétique**, avec un
cercle orange et un carré gris. L'observation est enregistrée dans
`data/evaluation/vision_observation.json`. Ce contrôle confirme l'exécution du pipeline,
pas une précision validée sur des photographies de maintenance. Le modèle est sensible
à la formulation du prompt et reste susceptible d'erreurs de couleur, d'OCR et de symboles.

La recette HTTP réelle passe aussi : authentification exigée, image invalide rejetée,
VLM exécuté et demande de clarification sans citation documentaire en l'absence de
référence. Durée observée : **39,94 secondes** sur cette machine. Rapport :
`data/evaluation/v2_photo_live.json`. Ce temps n'est pas une garantie de performance.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/evaluate_v2_photo.py
```

## État des livrables de l'étape 2

| Livrable | État |
| --- | --- |
| L2.1 — Analyse d'images avec VLM | Prototype local implémenté ; validation métier à faire |
| L2.2 — RAG multimodal | Fusion RRF puis reranking implémentés ; réponse photo fondée sur les textes des pages candidates autorisées |
| L2.3 — Jeu de données d'adaptation | Export, formulaire et validateur disponibles ; 800 à 1 500 paires métier vérifiées à constituer |
| L2.4 — Reranker adapté | Adaptation de la tête exécutée sur 24 paires fictives ; modèle de démonstration exclu de l'application |
| L2.5 — Comparaison nDCG@5 / MRR@10 | Comparaison exécutée sur candidats identiques et splits distincts ; validation métier à faire |
| L2.6 — Application texte + photo | Prototype V2 intégré ; recette métier ouverte |

L'adaptation métier nécessite des annotations vérifiées ; les fixtures de test et
la seule feuille de route ne sont pas un jeu d'entraînement métier. La démonstration
ne montre aucun gain du modèle adapté par rapport au modèle initial.

## Explorateur de figures — second incrément

Dans l'interface, cocher **« Explorer les figures et schémas associés »**, puis poser
une question. Une image peut être jointe pour ajouter la recherche visuelle.
La référence d'équipement facultative restreint les documents aux associations
déclarées dans le manifeste ; les droits de l'identité restent toujours appliqués.
Les résultats sont des correspondances à vérifier, pas des diagnostics.

Installation des deux encodeurs et indexation administrative :

```powershell
.\.venv\Scripts\python.exe scripts/download_clip.py
# API arrêtée :
.\.venv\Scripts\python.exe -m maintidoc.figures
.\.venv\Scripts\python.exe scripts/start_demo.py
```

Après ajout ou modification d'un PDF, réindexer d'abord le texte avec
`python -m maintidoc.ingest`, puis les figures. Le snapshot des figures doit
correspondre au snapshot textuel et aux révisions des encodeurs. Sinon l'API refuse
la recherche avec HTTP 409. Les anciens snapshots restent conservés sur disque.

### Extraction et stockage

- Les placements d'images incorporées sont extraits depuis leur région dans la page.
- Les tracés vectoriels proches sont regroupés par PyMuPDF et rendus comme régions.
- Les coordonnées sont exprimées en points PDF dans la page sans rotation ; la
  rotation originale est conservée dans les métadonnées de l'index.
- Les petites régions (moins de 12 points de côté ou 256 points carrés) et les grands
  arrière-plans vectoriels sont exclus. Des régions quasi identiques sont dédupliquées.
- Plus de 100 régions sur une page provoquent un refus d'indexation, sans activation
  d'un index partiel. Les PNG rendus ne dépassent pas 1 024 pixels de côté.
- Les figures conservent page, version, SHA-256 du PDF et du PNG, rôles et type de région.
- Les vecteurs sont mis en cache par empreinte d'image et révision d'encodeur.

Ces heuristiques ne constituent pas une segmentation intelligente : des logos,
tableaux et éléments décoratifs peuvent être inclus, et des figures peuvent être
partielles ou manquantes. Le corpus actuel produit **22 régions**, qui ne doivent pas
être présentées comme 22 schémas techniques de maintenance.

### Recherche et fusion

Les encodeurs sont `sentence-transformers/clip-ViT-B-32` pour les images et
`sentence-transformers/clip-ViT-B-32-multilingual-v1` pour les questions. Ils utilisent
un espace compatible de 512 dimensions. Les deux modèles sont chargés séquentiellement
dans des processus éphémères ; il n'y a pas de service externe d'inférence.

La fusion RRF additionne `1 / (60 + rang)` pour les pages uniques de chaque liste :
texte BGE, figures via texte CLIP, et figures via image CLIP si elle est fournie.
Deux passages de la même page ne donnent pas deux votes dans une liste. Les versions
de document ne sont pas mélangées. Les scores cosinus bruts ne sont pas additionnés.
Les résultats CLIP ne possèdent pas encore de seuil d'abstention calibré sur le métier :
une figure visuellement similaire peut être sans rapport technique avec la question.

Les endpoints `POST /search-multimodal` et `GET /figures/{id}/image` exigent le jeton
d'accès. La sélection est filtrée avant classement ; le téléchargement recontrôle
les droits et les empreintes des fichiers. Une figure ou un PDF modifié est refusé.

### Recettes

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_clip.py
.\.venv\Scripts\python.exe scripts/evaluate_multimodal.py
```

La première recette utilise les vrais modèles et un PDF synthétique séparé du corpus :
« un cercle orange » retrouve la page 1, « un carré bleu » la page 2 ; une requête
image retrouve aussi sa figure. Résultats dans `data/evaluation/v2_clip_live.json`.
Ce test sur deux formes ne constitue pas une évaluation métier ou un benchmark.
Un premier chargement à froid a dépassé le délai ; les recettes suivantes ont réussi.
Les durées observées ne sont pas garanties sur une autre charge ou machine.

La seconde recette vérifie l'API, la fusion et les téléchargements autorisés sur le
corpus actif, puis recherche une image identique à une région indexée. Rapport :
`data/evaluation/v2_multimodal_live.json`. Elle ne mesure pas la robustesse à un
changement de perspective ou à une photographie réelle d'équipement.

Recette du 23 octobre 2026 réussie : 22 régions dans l'index actif, aperçus PNG et PDF
source accessibles avec authentification, identifiant inconnu refusé. Temps observés :
77,92 s pour la recherche texte et 78,85 s pour texte + image sur cette machine.
La requête par image identique retrouve sa région avec une similarité supérieure à
0,99 ; cela vérifie la cohérence de l'index, pas la précision sur des photos nouvelles.

## Sources techniques

- Modèle, langue et utilisation : https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct
- Encodeur multilingue compatible CLIP envisagé : https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1
- Évaluation de rerankers : https://sbert.net/docs/package_reference/cross_encoder/evaluation.html
