"""Crée une démonstration explicite et un formulaire d'annotation du corpus réel."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maintidoc.adaptation import fingerprint, validate_dataset


def demo_dataset():
    # Faits inventés, uniquement pour vérifier entraînement, sauvegarde et métriques.
    examples = [
        ("Quelle est la couleur du voyant ALPHA ?", "Le voyant ALPHA de la machine fictive A est orange.",
         "Le voyant ALPHA n'indique pas la pression du circuit.", "Le voyant de la station fictive A clignote au démarrage."),
        ("Quel filtre utilise la pompe BETA ?", "La pompe fictive BETA utilise le filtre F42.",
         "La pompe BETA possède un carter métallique.", "Le filtre de la machine fictive B doit être inventorié."),
        ("Quelle tension alimente le capteur GAMMA ?", "Le capteur fictif GAMMA est alimenté en 24 volts.",
         "Le capteur GAMMA mesure la température ambiante.", "La tension du banc fictif C est consignée séparément."),
        ("Que signifie le code DELTA ?", "Le code DELTA de l'appareil fictif D signale une porte ouverte.",
         "Le code DELTA apparaît sur l'écran supérieur.", "La porte du local fictif D possède une serrure."),
        ("Où se trouve le fusible EPSILON ?", "Le fusible EPSILON du banc fictif E est dans le compartiment gauche.",
         "Le fusible EPSILON porte un repère sur son étiquette.", "Le compartiment droit du banc fictif E contient un câble."),
        ("Quel est le diamètre du joint ZETA ?", "Le joint ZETA de la vanne fictive F a un diamètre de 18 millimètres.",
         "Le joint ZETA est livré dans une pochette.", "Le diamètre du tuyau fictif F figure sur un autre plan."),
        ("Que signifie le voyant THETA ?", "Le voyant THETA du panneau fictif G indique un réservoir vide.",
         "Le voyant THETA est placé sur le panneau avant.", "Le réservoir fictif G possède un bouchon gris."),
        ("Quelle est la référence du roulement OMEGA ?", "Le roulement OMEGA du moteur fictif H a la référence R88.",
         "Le roulement OMEGA est placé derrière une protection.", "La référence du moteur fictif H est inscrite sur sa plaque."),
    ]
    queries = []
    for i, (question, positive, hard, negative) in enumerate(examples):
        candidates = [{"id": f"demo-{i}-{j}", "text": text, "retrieval_score": 0.9 - j * 0.1}
                      for j, text in enumerate([hard, positive, negative])]
        queries.append({"id": f"demo-q{i}", "group": f"fictif-{i}",
                        "split": "train" if i < 4 else ("dev" if i < 6 else "test"),
                        "query": question, "candidates": candidates,
                        "gold": {c["id"]: int(j == 1) for j, c in enumerate(candidates)},
                        "review": {"verified": False, "method": "synthetic_programmatic_ground_truth"}})
    return {"schema_version": 1, "demo_only": True,
            "notice": "Faits fictifs : aucune consigne de maintenance réelle, aucune annotation humaine.", "queries": queries}


def annotation_template(root):
    state = json.loads((root / "data/index_v1.json").read_text(encoding="utf-8"))
    documents = {d["id"]: d for d in state["documents"]}
    sources = [{"document_id": d["id"], "version": d["version"], "path": d["path"],
                "sha256": d["sha256"], "pages": d["pages"]} for d in documents.values()]
    return {"schema_version": 1, "demo_only": False, "queries": [], "available_sources": sources,
            "annotation_example": {"id": "A-REMPLACER", "group": "famille-equipement-document",
                "split": "train", "query": "Question réelle à annoter",
                "candidates": [{"id": "passage-1", "text": "Extrait exact de la page PDF, à remplacer.",
                                "retrieval_score": 0.0,
                                "source": {"document_id": "A-REMPLACER", "version": "A-REMPLACER",
                                           "path": "data/raw/manuel.pdf", "sha256": "A-REMPLACER", "page": 1}}],
                "gold": {"passage-1": None},
                "review": {"verified": False, "reviewer": "", "reviewed_at": ""}},
            "instructions": "Remplir queries avec 800–1500 paires vérifiées ; groupes ET documents disjoints entre train/dev/test. Le modèle d'exemple n'est pas un exemple validé."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", type=Path)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if args.validate:
        try:
            result = validate_dataset(json.loads(args.validate.read_text(encoding="utf-8")), demo=args.demo, root=ROOT)
        except (ValueError, OSError) as error:
            parser.exit(1, f"Validation refusée : {error}\n")
        print(json.dumps(result, indent=2))
    else:
        destination = ROOT / "data/adaptation"
        destination.mkdir(exist_ok=True)
        for name, data in [("demo.json", demo_dataset()), ("annotation_template.json", annotation_template(ROOT))]:
            path = destination / name
            if not path.exists():
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Démonstration synthétique et formulaire créés sans écraser les annotations existantes.")
