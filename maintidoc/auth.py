import hashlib
import hmac
import json

from maintidoc.schemas import Identity


def authenticate(token, root):
    if not token:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    users = json.loads((root / "config/users.local.json").read_text(encoding="utf-8"))["users"]
    for user in users:
        if user.get("enabled") is True and hmac.compare_digest(digest, user["token_sha256"]):
            return Identity(user_id=user["user_id"], roles=user["roles"])
    return None
