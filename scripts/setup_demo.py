"""Crée une identité locale de démonstration ; aucun jeton n'est affiché."""
import hashlib
import json
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[1]
users_path = root / "config/users.local.json"
token_path = root / "data/runtime/demo-token.txt"
if users_path.exists():
    print("Identités existantes conservées.")
else:
    token = secrets.token_urlsafe(32)
    users_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    users_path.write_text(json.dumps({"users": [{
        "user_id": "demo-technician", "roles": ["technician"], "enabled": True,
        "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
    }]}, indent=2), encoding="utf-8")
    token_path.write_text(token, encoding="utf-8")
    print("Identité créée. Jeton local : data/runtime/demo-token.txt (exclu du suivi Git).")
