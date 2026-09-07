"""Smoke-test an installed essentia wheel by decoding real audio files.

Run against an interpreter that already has the wheel installed, on a stock
distro image rather than the manylinux builder, so that the check exercises the
libraries vendored into the wheel and not the ones the build container happens
to have lying around:

    docker run --rm -v "$PWD":/w -w /w arm64v8/python:3.12-slim bash -c \
        'pip install wheelhouse/essentia-*-cp312-*aarch64.whl &&
         python packaging/verify_wheel_audio.py test/audio/recorded/*.mp3'
"""

import sys

import essentia.standard
import essentia.streaming  # noqa: F401  -- both bindings must import
from essentia.standard import Chromaprinter, MetadataReader, MonoLoader, YamlInput  # noqa: F401


def main(paths):
    if not paths:
        sys.exit("no audio files given; the shell glob matched nothing")

    for path in paths:
        audio = MonoLoader(filename=path, sampleRate=44100)()
        peak = float(abs(audio).max()) if len(audio) else 0.0
        print(f"{path}: {len(audio)} samples, peak {peak:.4f}")

        if len(audio) == 0:
            sys.exit(f"{path} decoded to zero samples")
        if peak == 0.0:
            sys.exit(f"{path} decoded to silence")

    print(f"OK: decoded {len(paths)} file(s) with essentia {essentia.__version__}")


if __name__ == "__main__":
    main(sys.argv[1:])
