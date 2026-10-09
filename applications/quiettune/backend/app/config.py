import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("QUIETTUNE_DATA", ROOT / "data")).resolve()
MODELS = Path(os.environ.get("QUIETTUNE_MODELS", ROOT / "models")).resolve()
FFMPEG = os.environ.get("QUIETTUNE_FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("QUIETTUNE_FFPROBE", "ffprobe")
NODE = os.environ.get("QUIETTUNE_NODE", "node")
DATA.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 200 * 1024 * 1024
MAX_SECONDS = 1200
EXTENSIONS = {".mp3", ".flac", ".wav", ".m4a", ".ogg"}
ANALYZER_VERSION = "quiettune-1.0.0"
DEFAULTS = {
    "weights": {"aggressive": 0.22, "unrelaxed": 0.15, "loudness": 0.08,
                "brightness": 0.23, "density": 0.22, "flatness": 0.10},
    "peak_mix": 0.25, "threshold": 60, "comfort_threshold": 35,
    "window_seconds": 5, "concurrency": 1, "ai_enabled": True,
}
