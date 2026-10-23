# Étape 2 — bilan du prototype technique

La chaîne technique est implémentée. **La recette métier de l'étape 2 reste ouverte** :
le corpus disponible est la feuille de route du projet, pas un ensemble de manuels
de maintenance. Les 800 à 1 500 paires vérifiées demandées ne sont pas disponibles.
Une démonstration synthétique ne remplit pas cette exigence et ne prouve pas une
amélioration sur la maintenance réelle.

## Livrables et preuves

| Livrable | Réalisation | Limite de validation |
| --- | --- | --- |
| L2.1 — VLM | SmolVLM-256M local, image normalisée, observations séparées des preuves | Petit modèle sensible aux couleurs, symboles et inscriptions ; recette métier à faire |
| L2.2 — RAG multimodal | Extraction des régions PDF, CLIP multilingue, fusion RRF puis reranking | 22 régions de la feuille de route ; aucune précision métier mesurée |
| L2.3 — Données | Export des passages, formulaire, schéma et validation des splits/provenances/revues | 12 passages non annotés ; 800–1500 paires humaines toujours à constituer |
| L2.4 — Adaptation | Tête d'un reranker multilingue entraînée sur CPU, sauvegardée et versionnée | 24 paires fictives ; adaptateur bloqué dans l'application |
| L2.5 — Comparaison | Sans reranker / initial / adapté, nDCG@5 et MRR@10, candidats identiques | Deux questions de test synthétiques ; pas de benchmark métier |
| L2.6 — Application | API et Streamlit texte/photo, citations, incertitudes, clarification | Prototype local, pas une qualification pour diagnostic ou action |

## Ce qui a été exécuté

Le reranker initial est `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, révision
`1427fd652930e4ba29e8149678df786c240d8825`. Le téléchargement est enregistré dans
`data/evaluation/reranker_model.json`, avec une empreinte SHA-256 des poids.
L'application utilise ce modèle initial par défaut.

L'adaptation a utilisé **8 questions fictives et 24 paires** : 12 paires train,
6 dev, 6 test. Les groupes, questions et passages sont distincts entre splits.
L'encodeur reste gelé ; seule la tête de classement apprend avec BCEWithLogitsLoss,
AdamW, graine 42, 8 époques et 32 mises à jour. Le checkpoint est choisi sur dev,
avec conservation du premier meilleur résultat ; le test est évalué ensuite.
Les poids ont effectivement changé et le fichier sauvegardé a été rechargé.

Résultats sur **2 questions de test fictives** :

| Configuration | nDCG@5 | MRR@10 |
| --- | ---: | ---: |
| Sans reranker | 0,6309 | 0,5000 |
| Reranker initial | 1,0000 | 1,0000 |
| Reranker adapté | 1,0000 | 1,0000 |

**Aucun gain de l'adaptation sur le modèle initial n'est démontré.** Ce jeu est trop
petit et artificiel pour une conclusion métier. La comparaison ne rajoute pas de
positifs aux candidats ; un positif absent du retrieval reste absent. Le gold
peut contenir des documents non retrouvés, pris en compte dans l'idéal du nDCG.
Les ex æquo sont départagés de manière déterministe.

Rapport : `data/evaluation/v2_reranker_demo.json`. L'adaptateur est dans le dossier
`.models/reranker-demo-…` indiqué par ce rapport. Les modèles ne sont jamais écrasés.
Les métriques de démonstration ne constituent pas une autorisation de l'activer.

Le contrôle indépendant `evaluate_reranker.py` a également rechargé l'adaptateur
dans de nouveaux processus et retrouvé exactement ces trois résultats.
La suite complète compte **38 tests réussis**, dont le filtrage avant reranking,
les métriques avec positifs absents, les fuites entre splits, la provenance PDF,
les empreintes des modèles et le refus d'activer un adaptateur fictif.
Les quatre cas HTTP documentaires passent avec le reranker initial : trois réponses
vérifiées et une abstention hors sujet. Les durées observées sont d'environ 50–53 s
par réponse et 22 s pour l'abstention ; elles ne constituent pas une garantie.
La recette HTTP de l'explorateur a aussi été relancée avec le reranker actif :
fusion, authentification, PNG et PDF autorisés, rejet des identifiants inconnus et
recherche d'une image identique réussis. Temps observés : 55,49 s en texte et
66,53 s en texte + image. Rapport : `data/evaluation/v2_multimodal_live.json`.

## Essayer l'application

Ouvrir http://127.0.0.1:8501. Une question sans image utilise BGE-M3, le reranker
initial, puis la sélection et la vérification des citations. Une photo passe d'abord
par le VLM. Sans référence exacte et manuel associé, une clarification est demandée.

Avec un équipement associé à un manuel autorisé, la recherche photo fusionne les
passages BGE et les correspondances CLIP. Les pages candidates peuvent apporter des
passages textuels supplémentaires : ils sont repris dans l'index textuel avec leurs
droits et doivent franchir le seuil BGE avant reranking. Seuls ces textes documentaires
peuvent être cités ; ni la légende VLM ni la similarité d'image ne sont des preuves.
La lecture est bornée aux cinq premières pages fusionnées et à 32 passages par page.

L'explorateur fusionne les listes puis reranke les pages disposant de texte. Une
figure sans texte conserve sa place RRF, sans description technique inventée.
Tous les modèles sont chargés dans des processus éphémères successifs pour limiter
la mémoire. Le CPU peut rendre une requête lente ; aucun délai n'est garanti.

## Reproduire la démonstration

Depuis le dossier Maintidoc, dans PowerShell :

```powershell
.\.venv\Scripts\python.exe scripts/download_reranker.py
.\.venv\Scripts\python.exe scripts/prepare_adaptation.py
.\.venv\Scripts\python.exe scripts/train_reranker.py --dataset data/adaptation/demo.json --demo
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/validate_stage2.py
```

Pour vérifier séparément le modèle sauvegardé, remplacer le chemin par celui du rapport :

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_reranker.py --dataset data/adaptation/demo.json --adapter .models/reranker-demo-VERSION --demo
```

`--demo` doit être explicite. Aucun fine-tuning du générateur de texte ou du VLM
n'a été effectué : le composant adapté est le reranker, conformément à la feuille de route.
La méthode disponible adapte la tête, pas tous les paramètres de l'encodeur.
`validate_stage2.py` produit `data/evaluation/v2_final_checks.json`, qui sépare les
contrôles techniques disponibles de la recette métier à réaliser. Les rapports
VLM et CLIP portent sur des dessins géométriques ; leurs limites restent explicites.
Des avertissements tiers SQLite/Qdrant et Streamlit peuvent apparaître dans les
tests ; ils n'ont pas empêché leur réussite sur cette machine.

## Constituer les données métier

1. Ajouter les vrais PDF autorisés et leurs associations aux équipements ; réindexer
   le texte puis les figures selon le guide V2.
2. Exécuter `scripts/export_annotation_pool.py`. L'export contient les passages,
   documents, pages, versions, droits et empreintes, **sans annotation inventée**.
3. Copier `data/adaptation/annotation_template.json` vers `metier.json`. Remplir
   `queries` avec des questions réelles et des listes de candidats issues du retrieval.
   Garder les candidats et leurs scores figés ; annoter leur pertinence de 0 à 3.
4. Réserver des groupes ET documents distincts pour train/dev/test. Ne pas répartir
   des passages du même manuel dans plusieurs splits. Inclure positifs, négatifs
   difficiles et négatifs non pertinents ; train exige les deux classes par question.
5. Faire vérifier les annotations par une personne habilitée, renseigner son nom,
   `verified: true` et la date ISO avec fuseau. Ces champs attestent une revue à réaliser ;
   le logiciel ne peut pas certifier qu'une revue humaine a effectivement eu lieu.
6. Valider, entraîner, comparer, puis effectuer la recette métier indépendante.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_adaptation.py --validate data/adaptation/metier.json
.\.venv\Scripts\python.exe scripts/train_reranker.py --dataset data/adaptation/metier.json
```

Le validateur refuse les jeux synthétiques en mode métier, les revues absentes,
les passages non vérifiables dans le PDF actif, les empreintes ou versions changées,
les fuites entre splits et un total hors de 800–1500 paires. Les PDF sans couche texte
exploitable peuvent nécessiter une version OCR contrôlée avant cette vérification.
Les données et adaptateurs restent locaux et sont exclus de Git par défaut.

## Configuration et suite

`MAINTIDOC_RERANKER` accepte `off`, `initial` (par défaut) ou `adapte`.
Le dernier mode exige `MAINTIDOC_RERANKER_ADAPTER` pointant vers un adaptateur local
dans `.models`, non marqué démonstration, avec métadonnées et empreintes cohérentes.
Une sélection manuelle reste nécessaire après évaluation : aucun changement de modèle
n'est décidé automatiquement d'après les métriques.

L'étape 3 pourra ajouter l'historique via MCP et les brouillons de tickets. La création
effective d'un ticket devra toujours attendre la validation humaine de son contenu.
Ces fonctions ne sont pas encore présentes dans le prototype V2.

Sources : [modèle multilingue](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1),
[entraînement des cross-encoders](https://www.sbert.net/docs/cross_encoder/training_overview.html),
[évaluation des rerankers](https://sbert.net/docs/package_reference/cross_encoder/evaluation.html).
