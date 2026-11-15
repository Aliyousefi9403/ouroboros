"""Export trained weights for the browser:  python scripts/export_web.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ouroboros.models import MODEL_DIR, export_web  # noqa: E402

out = os.path.join(os.path.dirname(MODEL_DIR), "docs", "models")
os.makedirs(out, exist_ok=True)
export_web(os.path.join(MODEL_DIR, "reflex.npz"), out, "reflex", {"kind": "mlp", "inputs": 11})
export_web(os.path.join(MODEL_DIR, "board.npz"), out, "board", {"kind": "cnn", "size": 10})
print("exported to", out)
