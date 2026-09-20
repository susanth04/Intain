"""Load backend/.env so OPENAI_API_KEY is available to uvicorn."""
from pathlib import Path
import os

_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent


def load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_BACKEND_DIR / ".env", override=False)
        load_dotenv(_ROOT_DIR / ".env", override=False)
        return
    except Exception:
        pass
    for path in (_BACKEND_DIR / ".env", _ROOT_DIR / ".env"):
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and not os.environ.get(key):
                os.environ[key] = value
