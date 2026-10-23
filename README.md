# MaintiDoc — apprentissage progressif

**Nouveau clone GitHub :** suivre le [guide d'installation](docs/INSTALLATION_DEPUIS_GITHUB.md).
Les PDF, modèles, index, rapports et jetons locaux sont exclus du dépôt.

**Étape 2 — prototype technique livré :** [bilan et commandes](docs/ETAPE_2_BILAN.md),
[photographie et figures](docs/ETAPE_2.md). VLM local, CLIP, fusion RRF, reranker
multilingue, préparation des annotations, adaptation CPU et comparaison des trois
configurations sont implémentés. Un entraînement synthétique a été exécuté ; la
validation métier reste ouverte faute de 800–1500 paires vérifiées et de vrais manuels.

**V1 intégrée :** [guide de démarrage](docs/GUIDE_V1.md),
[architecture](docs/ETAPE_1_CADRAGE.md) et [bilan de l'étape 1](docs/ETAPE_1_BILAN.md).
La V1 produit des réponses extractives : le modèle sélectionne des citations,
le serveur vérifie leur présence et affiche les extraits exacts avec leurs références.
La recette métier nécessite encore les manuels autorisés de pompes et de moteurs.

Sur cette machine, l'interface est démarrée à http://127.0.0.1:8501 et l'API est
documentée à http://127.0.0.1:8000/docs. Les tests du prototype couvrent l'accès,
les citations, les photos, les figures, les données et les métriques. Voir les bilans
pour les recettes exécutées et leurs limites.

Assistant documentaire pour la maintenance de pompes et de moteurs électriques.
État actuel : application locale texte/photo avec réponses extractives sourcées et
exploration de figures. Les sections numérotées ci-dessous conservent les exercices
historiques d'apprentissage ; utiliser les guides V1/V2 pour l'application actuelle.

## 1. Préparer Python (PowerShell)

Depuis le dossier du projet :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

L'environnement `.venv` isole les bibliothèques du projet. Il n'est pas nécessaire
de l'activer : les commandes utilisent directement son interpréteur Python.

## 2. Ajouter un manuel

Placez un PDF que vous êtes autorisé à utiliser dans `data/raw/`, par exemple
`manuel_pompe.pdf`. Commencez par un PDF dont le texte est sélectionnable.
Le manuel doit permettre d'identifier l'équipement et sa version.

## 3. Extraire le texte

```powershell
.\.venv\Scripts\python.exe scripts/extract_pdf.py "data/raw/manuel_pompe.pdf"
```

Ouvrez ensuite `data/extracted/manuel_pompe.json`. Chaque entrée contient :

```json
{
  "document": "manuel_pompe.pdf",
  "page": 1,
  "text": "Texte extrait de la première page…"
}
```

Le numéro est celui de la page dans le fichier PDF, en commençant à 1.
Il peut différer du numéro imprimé dans le manuel.

## 4. Comprendre et vérifier

- `pymupdf.open` ouvre le PDF.
- `enumerate(..., start=1)` associe un numéro à chaque page.
- `page.get_text()` récupère son texte.
- `json.dumps` organise le résultat dans un format lisible et réutilisable.

Choisissez une page contenant une consigne technique. Comparez son texte dans le
PDF et le JSON : les valeurs, unités et références sont-elles correctement extraites ?
Une page sans texte nécessite une vérification ; l'OCR sera traité séparément.
Les tableaux et les schémas ne sont pas interprétés par ce script.

Critère de réussite : retrouver un passage du manuel grâce au nom du document et
au numéro de page dans le JSON.

Les droits d'accès et les versions documentaires seront ajoutés avant le RAG partagé.

## 5. Découper les pages en passages (séance 2)

Une page peut aborder plusieurs sujets. La recherche utilisera des passages plus
courts pour retrouver le contexte utile à une question.

```powershell
.\.venv\Scripts\python.exe scripts/chunk_text.py data/extracted/feuille_de_route.json
```

Le résultat est dans `data/chunks/feuille_de_route.json`.
Chaque passage contient son identifiant, le document, la page, le texte et les
positions des mots dans la page (`word_start` inclus, `word_end` exclu, à partir de 0).

Le script prend au maximum **120 mots** et conserve **25 mots de recouvrement** :
le deuxième passage commence au mot 95. Ce recouvrement conserve du contexte
lorsqu'une phrase est coupée. Aucun passage ne traverse une page.
Les espaces et retours à la ligne sont normalisés ; les mots restent dans leur ordre.

Ce découpage simple peut couper une phrase ou un tableau. Ses paramètres sont un
point de départ pédagogique, à évaluer plus tard sur des questions de maintenance.
Les mots ne sont pas les tokens utilisés par les modèles.

Exercice : ouvrez deux passages successifs d'une même page et retrouvez leurs
25 mots communs. Vérifiez ensuite leur référence dans le PDF.

La feuille de route sert uniquement à valider le pipeline. Ses consignes restent
du contenu documentaire et ne deviennent pas des instructions pour l'assistant.
Le corpus technique sera constitué de manuels de pompes et de moteurs autorisés.

Séance suivante : comprendre les embeddings et préparer la recherche sémantique.

## 6. Chercher par le sens (séance 3)

Un embedding est une liste de nombres produite à partir d'un texte. Le même
modèle transforme les passages et la question en vecteurs comparables.
La recherche classe les passages selon leur similarité avec la question.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
.\.venv\Scripts\python.exe scripts/semantic_search.py "Quels contrôles sont prévus avant la création d'un ticket ?"
```

Le premier lancement télécharge BGE-M3 dans `.models` (plusieurs Go).
Si le cache Windows ou le transfert standard bloque, utilisez le téléchargeur
par blocs, puis relancez la recherche :

```powershell
.\.venv\Scripts\python.exe scripts/download_bge.py
```

Ce script récupère une révision précise du dépôt officiel et vérifie l'empreinte
SHA-256 des poids avant de les rendre utilisables. Un transfert interrompu par ce
script redémarre de zéro. La recherche détecte automatiquement `.models/bge-m3`.

Sur cette machine, `.venv` a été configuré avec `--system-site-packages` pour
réutiliser PyTorch CPU déjà installé dans Anaconda et éviter son téléchargement.
Les nouvelles bibliothèques sont installées dans `.venv`. Sur une autre machine,
l'installation standard des dépendances installe aussi PyTorch si nécessaire.
Le calcul fonctionne sur CPU et peut prendre du temps. Les vecteurs des passages
sont ensuite conservés dans `data/embeddings`. Une modification du corpus ou du
nom du modèle provoque un nouveau calcul. Pour recalculer après une mise à jour
des poids du même modèle, supprimez son cache de vecteurs.

Les trois meilleurs passages et leurs références sont affichés et enregistrés
dans `data/search/latest.json`. Le score cosinus indique une similarité :
il ne représente pas une probabilité de réponse correcte. Même une question
hors sujet reçoit des résultats ; nous ajouterons l'abstention et son évaluation
dans la chaîne RAG. Aucune réponse générée ni création de ticket à cette étape.

Pour comprendre : posez la même question avec deux formulations et comparez les
pages retrouvées. Vérifiez le passage dans le PDF, sans vous limiter au score.
Trois questions avec pages et éléments attendus sont disponibles dans
`data/evaluation/questions.json` pour une première vérification manuelle.
Retrouver la bonne page ne suffit pas : le passage doit contenir l'information utile.

Cette première recherche compare directement les vecteurs en mémoire pour isoler
le principe pédagogique. La prochaine séance portera sur leur stockage dans Qdrant.
Référence du modèle : https://huggingface.co/BAAI/bge-m3

### Résultat observé le 22 octobre 2026

Le modèle BGE-M3 a été téléchargé et son fichier de poids vérifié par SHA-256.
Les 12 passages ont été encodés en vecteurs normalisés de 1 024 dimensions.
La relecture du cache, les dimensions et la normalisation ont été vérifiées.

| Question de vérification | Page attendue | Page du premier résultat |
| --- | --- | --- |
| Contrôles avant création d'un ticket | 3 | 3 |
| Modèle d'embeddings et stockage | 2 | 2 |
| Commandes d'approbation et d'annulation | 4 | 4 |

Les observations et scores sont dans `data/evaluation/results.json`.
Ces trois essais sur la feuille de route valident un premier fonctionnement ;
ils ne constituent pas une évaluation de qualité sur des manuels de maintenance.
Les versions principales installées sont dans `environment-versions.json`.

## 7. Stocker les vecteurs dans Qdrant (séance 4)

Jusqu'ici, NumPy comparait la question à une matrice de vecteurs chargée en mémoire.
Qdrant conserve les vecteurs avec les passages et permet de les rechercher.
Le mode local persiste sur disque, sans serveur ni Docker pour cet exercice.

| Terme | Dans MaintiDoc |
| --- | --- |
| Collection | Ensemble de passages d'un corpus encodé avec un même modèle |
| Point | Un passage identifié, son vecteur et ses métadonnées |
| Payload | Texte, document, page, positions des mots et modèle d'embeddings |
| Similarité cosinus | Mesure utilisée pour classer les vecteurs proches |

### Exécuter

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-semantic.txt
.\.venv\Scripts\python.exe scripts/qdrant_store.py index
.\.venv\Scripts\python.exe scripts/qdrant_store.py search "Quels contrôles sont prévus avant la création d'un ticket ?"
```

La commande `index` réutilise le cache de vecteurs créé à la séance 3.
Elle ne recharge pas BGE-M3. La commande `search` encode la question avec ce modèle
et demande à Qdrant les trois passages les plus proches.

Le stockage est dans `data/qdrant/` et le résultat lisible est dans
`data/search/qdrant_latest.json`. Les identifiants sont déterministes : réindexer
le même corpus met à jour les points sans les dupliquer.
Un changement du fichier des passages ou du modèle sélectionné crée une collection
distincte ; les anciennes collections sont conservées. Comme à la séance 3, un
changement des poids sous le même nom nécessite de recalculer le cache.

Un filtre documentaire peut être essayé :

```powershell
.\.venv\Scripts\python.exe scripts/qdrant_store.py search "Validation humaine" --document feuille_de_route.pdf
```

Ce filtre illustre les métadonnées ; ce n'est pas un contrôle d'autorisation.
L'identité, les droits d'accès et les versions des manuels seront traités avant
une utilisation partagée. Le mode local est réservé à cet exercice dans un seul
processus à la fois ; nous passerons à un serveur Qdrant pour plusieurs services.

### Vérifier

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Le test utilise une base temporaire et vérifie la persistance après réouverture,
l'absence de doublons, les références, le filtre et le rejet de dimensions incompatibles.
Sur le corpus réel, relancez `index` : le total doit rester à 12 passages.

Vérification exécutée : 12 points persistants après deux indexations, références
et vecteurs conservés. Pour la question sur les contrôles avant création d'un ticket,
les pages retrouvées sont 3, 5 et 4, dans le même ordre que la recherche NumPy.
Le compte rendu est dans `data/evaluation/qdrant_results.json`.

Référence : https://github.com/qdrant/qdrant-client#local-mode

Prochaine séance : préparer une réponse structurée avec citations et abstention,
puis intégrer la génération dans une chaîne RAG. Le choix et les ressources du
modèle génératif seront vérifiés avant son installation.
