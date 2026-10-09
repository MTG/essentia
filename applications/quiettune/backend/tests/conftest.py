import os
import tempfile
from pathlib import Path

test_root = Path(__file__).resolve().parents[2] / ".Codex" / "test-data"
test_root.mkdir(parents=True, exist_ok=True)
os.environ["QUIETTUNE_DATA"] = tempfile.mkdtemp(prefix="quiettune-tests-", dir=test_root)
