"""Lance l'API et Streamlit en arrière-plan sur la boucle locale uniquement."""
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

root = Path(__file__).resolve().parents[1]
runtime = root / "data/runtime"
runtime.mkdir(parents=True, exist_ok=True)
if not (root / "data/index_v1.json").exists() or not (root / "config/users.local.json").exists():
    raise SystemExit("Exécutez d'abord scripts/setup_demo.py puis python -m maintidoc.ingest.")
for name, url, command in [
    ("api", "http://127.0.0.1:8000/health", [sys.executable, "-m", "uvicorn", "maintidoc.api:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "1"]),
    ("ui", "http://127.0.0.1:8501/_stcore/health", [sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py", "--server.address", "127.0.0.1", "--server.port", "8501", "--server.headless", "true", "--browser.gatherUsageStats", "false"]),
]:
    try:
        with urllib.request.urlopen(url, timeout=2):
            print(f"{name} répond déjà ; processus conservé.")
    except OSError:
        with (runtime / f"{name}.out.log").open("a") as out, (runtime / f"{name}.err.log").open("a") as err:
            process = subprocess.Popen(command, cwd=root, stdout=out, stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        (runtime / f"{name}.pid").write_text(str(process.pid), encoding="ascii")
        print(f"{name} lancé (PID {process.pid}).")
print("Interface : http://127.0.0.1:8501 — API : http://127.0.0.1:8000/docs")
