# Bilan de l'étape 1 — MaintiDoc V1

## Livrables

| Référence | Livrable réalisé | Emplacement |
| --- | --- | --- |
| L1.1 | Cadrage et architecture | `docs/ETAPE_1_CADRAGE.md` |
| L1.2 | Pipeline PDF, OCR optionnel et découpage | `maintidoc/ingest.py` |
| L1.3 | Qdrant, vecteurs et métadonnées d'accès | `data/qdrant_v1/`, `data/index_v1.json` |
| L1.4 | Chaîne LangChain et API FastAPI | `maintidoc/rag.py`, `maintidoc/api.py` |
| L1.5 | Interface de démonstration Streamlit | `ui/streamlit_app.py` |
| L1.6 | Tests documentaires et d'accès | `tests/`, `data/evaluation/v1_live.json` |

## Validation automatisée

Quatorze tests passent sur des fixtures synthétiques isolées. Ils vérifient :

- la persistance Qdrant et l'absence de doublons ;
- la conservation des pages, textes, versions et rôles ;
- le filtrage des passages avant génération ;
- le refus d'accès aux PDF non autorisés ;
- l'authentification et la désactivation d'un compte ;
- le refus de rôles envoyés par le client et des questions invalides ;
- l'abstention quand aucune source pertinente n'est accessible ;
- le rejet des citations inventées et des sorties mal formées ;
- l'erreur explicite 503 lorsque le générateur est indisponible ;
- la distinction entre instructions système et contenu documentaire dans la requête au modèle ;
- le rendu de la réponse et des citations dans Streamlit, et l'affichage d'une session refusée ;
- la conservation des accents dans le transfert vers le processus d'embeddings Windows.

Le test de séparation des instructions vérifie la construction de la requête, pas une résistance générale
à toutes les injections de prompt. Aucun outil d'action n'est exposé dans V1.
Le client Qdrant local émet un `ResourceWarning` SQLite pendant un test de stockage ;
les fermetures explicites, réouvertures et assertions de persistance passent.

Les tests Streamlit utilisent un transport HTTP simulé ; la recette réelle séparée
utilise les services et modèles locaux. Le message `missing ScriptRunContext` émis
par AppTest pendant l'exécution de tests n'est pas une erreur de l'interface servie.

## Recette réelle

Corpus : un PDF de cinq pages, découpé en douze passages. Modèles : BGE-M3 et
Qwen2.5 1.5B réellement installés et exécutés localement. Le script
`scripts/evaluate_v1.py` teste trois questions documentaires, une question hors corpus
et l'absence d'authentification. Les sorties et durées observées sont conservées dans
`data/evaluation/v1_live.json`. Le statut final est à lire dans `all_passed`.
La vérification exige les informations utiles dans les citations (approbation,
BGE-M3 et Qdrant, `/approve` et `/cancel`), et pas seulement la bonne page.

Résultat final observé le 22 octobre 2026 : **4 cas réels réussis**, accès sans
identité refusé avec HTTP 401, API et Streamlit accessibles avec HTTP 200.

| Cas réel | Résultat | Durée observée |
| --- | --- | --- |
| Contrôles avant création d'un ticket | Réponse avec preuve d'approbation | 27,36 s |
| Modèle d'embeddings et stockage | BGE-M3 et Qdrant présents dans les preuves | 27,97 s |
| Commandes du bot | `/approve` et `/cancel` présents dans les preuves | 30,99 s |
| Capitale de l'Australie, hors corpus | Abstention | 19,31 s |

Le PDF a été téléchargé depuis l'endpoint authentifié et les **9 citations** des
trois réponses ont été vérifiées directement dans le texte de leurs pages PDF.
Les temps ci-dessus sont des observations sur cette machine, pas des garanties.
Les empreintes et versions des modèles sont dans `data/evaluation/model_manifest_v1.json`.

Le seuil de démonstration a été ajusté à 0,40 après mesure des scores : environ
0,429 pour la question pertinente la moins proche, contre 0,265 pour la question
hors corpus. Ces quatre questions ont servi au réglage, puis à la vérification ;
ce ne sont pas un jeu d'évaluation indépendant. Voir `data/evaluation/v1_calibration.json`.

## Ce qui reste pour la recette métier

La base technique de V1 est livrée. **L'étape 1 ne peut pas être déclarée entièrement
validée au sens de la feuille de route avec le seul PDF de projet.** Il manque :

1. 30 à 50 documents métier autorisés sur deux familles d'équipements ;
2. des questions techniques avec réponses et pages de référence validées par un technicien ;
3. une évaluation sur ce corpus, notamment des abstentions et des droits réels ;
4. des essais OCR si le corpus contient des scans.

Les documents synthétiques des tests ne sont pas comptés comme des manuels métier.
Le dépôt Git n'a pas été initialisé ni de commit créé au nom de l'utilisateur.
Les fichiers sources et les contraintes de versions sont présents pour leur versionnement.

## Passage à V2

Le code est organisé pour ajouter l'analyse de photographies, l'indexation des figures
et le reranking sans contourner les contrôles d'accès et de citations. La sélection
du VLM et du reranker devra tenir compte de la mémoire disponible ; aucun modèle
multimodal ni fine-tuning n'est présenté comme déjà réalisé.
