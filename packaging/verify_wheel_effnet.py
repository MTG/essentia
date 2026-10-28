"""Smoke-test an installed essentia-tensorflow wheel by running TensorFlow inference.

The essentia-tensorflow wheels vendor the TensorFlow C library, so nothing else
has to be installed for a TensorFlow-backed algorithm to run. That the vendored
copy is present, has the right architecture, and is actually reachable from the
extension is not visible until one of those algorithms runs, so this loads a
real graph and checks the output.

Run against an interpreter that has the wheel installed, on a stock image or a
clean venv rather than the build machine, so that the check exercises what the
wheel ships and not what the builder happened to have lying around:

    docker run --rm -v "$PWD":/w -w /w python:3.12-slim bash -c \
        'pip install wheelhouse/essentia_tensorflow-*-cp312-*x86_64.whl &&
         python packaging/verify_wheel_effnet.py discogs-effnet-bs64-1.pb'

The audio is synthesised rather than decoded so that FFmpeg is not part of the
measurement; packaging/verify_wheel_audio.py covers the decoding path.

This is also cibuildwheel's per-interpreter test-command (see
cibuildwheel-tensorflow.toml), where it runs against every built wheel -- not
just the two checked afterwards in a clean container -- with the model shipped
in via test-sources. The model file at that call site is staged by the
workflow (build_wheels job, before cibuildwheel runs) behind an actions/cache
keyed on the model's SHA256; on a cold cache hitting a dead model host, the
workflow stages an empty placeholder instead of failing the build. A missing
or empty model here is therefore not an error: the import, vendored-library
and dependency-metadata checks below still run (they need no model and no
network), only the inference step is skipped, with a loud warning so the gap
is visible in the log rather than silently absent.

Two contract checks run unconditionally, independent of the model:

  - the installed distribution vendors a TensorFlow C library (and its
    "framework" companion) rather than depending on the `tensorflow` PyPI
    package -- auditwheel places it at <dist-name>.libs/ next to the top-level
    packages on Linux; delocate places it inside the package itself, at
    essentia/.dylibs/, on macOS.
  - the distribution's metadata declares no `tensorflow` requirement, which is
    the whole point of vendoring it: a regression to either contract (the
    library silently dropped, or a `tensorflow` dependency silently added
    back) is caught here rather than downstream at import or install time.
"""

import importlib.metadata
import os
import pathlib
import platform
import re
import sys

import numpy as np

import essentia
from essentia.standard import TensorflowPredictEffnetDiscogs

DISTRIBUTION_NAME = 'essentia-tensorflow'

# discogs-effnet expects 16 kHz mono. Ten seconds is several patches, so the
# output has more than one row and a wrong patch/hop configuration shows up.
SAMPLE_RATE = 16000
DURATION_SECONDS = 10

# The embeddings head, PartitionedCall:1, rather than the 400-way style
# classifier on PartitionedCall:0.
EMBEDDINGS_OUTPUT = 'PartitionedCall:1'
EMBEDDINGS_SIZE = 1280

# A real download is ~18 MB; anything under this is treated as "no model" --
# either the path does not exist, or the workflow staged an empty placeholder
# because the fetch failed or the checksum did not match.
MIN_MODEL_BYTES = 1_000_000


def loud_warning(message):
    """Print a warning that is hard to miss in a scrolling CI log."""
    banner = '!' * 70
    print(banner)
    print(f'WARNING: {message}')
    print(banner)
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        print(f'::warning::{message}')


def model_available(graph_filename):
    path = pathlib.Path(graph_filename)
    return path.is_file() and path.stat().st_size >= MIN_MODEL_BYTES


# A requirement string's project name is everything before the first of a
# version specifier, an extras marker, or an environment marker, e.g.
# "tensorflow" out of "tensorflow>=2.10; extra == 'tf'" or "tensorflow[and-cuda]".
_REQUIREMENT_NAME = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)')


def check_no_tensorflow_dependency():
    """The wheel vendors libtensorflow; it must not also depend on the pip package."""
    requires = importlib.metadata.requires(DISTRIBUTION_NAME) or []
    tf_requires = []
    for requirement in requires:
        match = _REQUIREMENT_NAME.match(requirement)
        name = match.group(1) if match else requirement
        if name.replace('_', '-').lower() == 'tensorflow':
            tf_requires.append(requirement)
    if tf_requires:
        sys.exit(f'{DISTRIBUTION_NAME} declares a tensorflow dependency: {tf_requires} '
                  '(it vendors libtensorflow and must not also depend on the pip package)')
    print(f'OK: {DISTRIBUTION_NAME} declares no tensorflow dependency '
          f'({len(requires)} requirement(s) total)')


def check_vendored_libtensorflow():
    """Assert the installed package vendors libtensorflow (+ framework)."""
    package_dir = pathlib.Path(essentia.__file__).resolve().parent
    site_packages = package_dir.parent
    system = platform.system()

    if system == 'Darwin':
        libdir = package_dir / '.dylibs'
    else:
        dist = importlib.metadata.metadata(DISTRIBUTION_NAME)
        libs_dirname = dist['Name'].replace('-', '_') + '.libs'
        libdir = site_packages / libs_dirname

    if not libdir.is_dir():
        sys.exit(f'expected vendored-library directory {libdir} not found; '
                  f'{DISTRIBUTION_NAME} does not appear to vendor libtensorflow')

    names = sorted(p.name for p in libdir.iterdir() if p.is_file())
    core = [n for n in names if n.lower().startswith('libtensorflow') and 'framework' not in n.lower()]
    framework = [n for n in names if n.lower().startswith('libtensorflow') and 'framework' in n.lower()]

    if not core:
        sys.exit(f'no vendored libtensorflow core library found in {libdir} (found: {names})')
    if not framework:
        sys.exit(f'no vendored libtensorflow_framework library found in {libdir} (found: {names})')

    print(f'OK: vendored {core[0]} and {framework[0]} under {libdir}')


def synthesise():
    """Return a deterministic 10 s 16 kHz mono signal as float32."""
    t = np.arange(SAMPLE_RATE * DURATION_SECONDS, dtype=np.float32) / SAMPLE_RATE
    # A couple of tones plus a slow sweep, so the spectrum is not a single line.
    signal = (0.4 * np.sin(2 * np.pi * 440.0 * t)
              + 0.3 * np.sin(2 * np.pi * 1320.0 * t)
              + 0.2 * np.sin(2 * np.pi * (200.0 + 60.0 * t) * t))
    return np.ascontiguousarray(signal, dtype=np.float32)


def run_inference(graph_filename):
    model = TensorflowPredictEffnetDiscogs(graphFilename=graph_filename,
                                           output=EMBEDDINGS_OUTPUT)
    embeddings = model(synthesise())

    print(f'embeddings {embeddings.shape} {embeddings.dtype}')
    print(f'first values {np.array2string(embeddings[0][:5], precision=5)}')

    if embeddings.ndim != 2:
        sys.exit(f'expected a 2-D array, got {embeddings.ndim} dimensions')
    if embeddings.shape[0] < 1:
        sys.exit('the model produced no patches')
    if embeddings.shape[1] != EMBEDDINGS_SIZE:
        sys.exit(f'expected {EMBEDDINGS_SIZE} features per patch, '
                 f'got {embeddings.shape[1]}')
    if embeddings.dtype != np.float32:
        sys.exit(f'expected float32 embeddings, got {embeddings.dtype}')
    if not np.isfinite(embeddings).all():
        sys.exit('the embeddings contain NaN or inf')

    print(f'OK: TensorflowPredictEffnetDiscogs returned {embeddings.shape}')


def main(argv):
    if len(argv) != 1:
        sys.exit('usage: verify_wheel_effnet.py <discogs-effnet-bs64-1.pb>')

    graph_filename = argv[0]

    print(f'essentia {essentia.__version__}')

    check_no_tensorflow_dependency()
    check_vendored_libtensorflow()

    if not model_available(graph_filename):
        loud_warning(f'{graph_filename!r} is missing or empty (< {MIN_MODEL_BYTES} bytes); '
                      'skipping TensorFlow inference and running import/vendoring/dependency '
                      'checks only. This means the model host was unreachable when the '
                      'workflow staged it -- see the "Fetch the discogs-effnet model" step.')
        print('OK: import + vendoring + dependency checks passed (inference skipped, no model)')
        return

    run_inference(graph_filename)


if __name__ == '__main__':
    main(sys.argv[1:])
