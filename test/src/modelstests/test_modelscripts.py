import subprocess
import sys
import unittest
from pathlib import Path


ESSENTIA_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(ESSENTIA_ROOT / "src" / "examples" / "python" / "models"))

from generate_example_scripts import generate_example_scripts


class TestModels(unittest.TestCase):
    output_dir = ESSENTIA_ROOT / "build" / "model_scripts"
    audio_file = ESSENTIA_ROOT / "test" / "audio" / "recorded" / "vignesh.wav"

    scripts = generate_example_scripts(
        output_dir,
        audio_file=str(audio_file),
        force=True,
        download_models=True,
    )

    def testScriptExecutes(self):
        for script in self.scripts:
            with self.subTest(i=script):
                subprocess.check_call(["python3", script], cwd=str(Path(script).parent))


if __name__ == "__main__":
    unittest.main()
