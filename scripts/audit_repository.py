"""Contrôle le contenu suivi/stagé avant publication, sans afficher de secret."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
paths = [p for p in result.stdout.decode("utf-8").split("\0") if p]
token_path = ROOT / "data/runtime/demo-token.txt"
local_token = token_path.read_bytes().strip() if token_path.exists() else b""
forbidden = (".models/", ".venv/", ".tools/", "data/runtime/", "data/qdrant", "data/figures_v2/")
patterns = [rb"gh[pousr]_[A-Za-z0-9_]{30,}", rb"github_pat_[A-Za-z0-9_]{30,}",
            rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
issues, size = [], 0
for name in paths:
    path = ROOT / name
    if (name.startswith(forbidden) or name in {"config/users.local.json", "config/corpus.json"}
            or (name.startswith("data/") and not name.endswith("/.gitkeep"))
            or path.suffix.lower() in {".pdf", ".bin", ".safetensors", ".pem", ".key"}
            or path.name == ".env" or path.name.startswith(".env.") and path.name != ".env.example"):
        issues.append(f"Fichier local exclu attendu : {name}")
        continue
    content = path.read_bytes()
    size += len(content)
    if len(content) > 5 * 1024 * 1024:
        issues.append(f"Fichier trop volumineux : {name}")
    if (local_token and local_token in content) or any(re.search(pattern, content) for pattern in patterns):
        issues.append(f"Secret potentiel détecté : {name}")
if not paths:
    issues.append("Aucun fichier suivi : exécuter git add avant le contrôle.")
if issues:
    raise SystemExit("\n".join(issues))
print(f"Contrôle réussi : {len(paths)} fichiers, {size:,} octets ; aucun secret détecté par ces contrôles.")
