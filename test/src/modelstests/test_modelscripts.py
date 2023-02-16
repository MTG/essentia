import subprocess
import sys
import unittest
from pathlib import Path


ESSENTIA_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.append(str(ESSENTIA_ROOT / "src" / "examples" / "python" / "models"))
sys.path.append(str(ESSENTIA_ROOT / "test" / "src" / "unittests"))

from essentia_test import testdata
from genereate_example_scripts import generate_example_scripts


class TestModels(unittest.TestCase):
    output_dir = ESSENTIA_ROOT / "build" / "script_models"
    audio_file = str(Path(testdata.audio_dir, "recorded", "vignesh.wav"))

    scripts = generate_example_scripts(
        output_dir,
        audio_file=audio_file,
        force=True,
        download_models=True,
    )

    def testScriptExecutes(self):
        for script in self.scripts:
            with self.subTest(i=script):
                subprocess.run(["python3", script], check=True)


if __name__ == "__main__":
    unittest.main()
