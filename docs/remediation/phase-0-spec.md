# Phase 0 — Implementation Spec: Validation scaffolding

**Objective:** build the test/measurement infrastructure that every later phase's acceptance gate depends on. **No
product code changes** — this phase adds test helpers, harness scripts, and CI lanes only. It must merge first so that
"green" and "byte-identical" and "RSS-stable" are objectively measurable before any behavioral fix lands.

**Phase models:** Code **Sonnet 5** / Review **Opus 4.8**. Low ambiguity; the reviewer mainly checks that the oracles
actually fail when they should (a harness that can't detect the bug it guards is worse than none).

**Branch:** `test/remediation-scaffolding`. Three independent checkpoints, mergeable in any order.

**Why this exists:** Batches B, C, and G change numerical output or touch threading/UB; they can only be validated
against external references (numpy, ffmpeg) and resource/thread instrumentation. Standing that up per-phase would be
duplicated and inconsistent — centralize it here.

Existing entry point (verified): `python test/src/unittests/all_tests.py <categories...>` with per-category test files
under `test/src/unittests/<category>/test_*.py`.

---

## CP0.1 — Baseline capture harness
**Branch:** `test/baseline-capture`  ·  Supports Phases 2, 3, 6

### Deliverable
A script (e.g. `test/remediation/capture_baselines.py`) that:
- Runs the full gate suite and records pass/fail per test:
  `python test/src/unittests/all_tests.py io base standard rhythm highlevel machinelearning audioproblems`.
- For the algorithms whose output later phases intentionally move, captures **numeric** baselines (arrays serialized
  to `.npy`) against the current `master`:
  - `ChromaCrossSimilarity`, `CrossSimilarityMatrix` (Phase 2 / BUG-01)
  - `FrameCutter` streaming with `startFromZero=false`, `silentFrames="noise"` (Phase 2 / BUG-05)
  - `TempoTapDegara`, `StochasticModelSynth`, `AudioOnsetsMarker` (Phase 6 / BUG DESIGN-05)
  - `AudioLoader` full decodes for one representative file per format family (Phase 3)
- Writes baselines to a versioned dir (`test/remediation/baselines/<gitsha>/`) with a manifest recording the source
  commit, so a phase can diff "before" vs "after" and prove *only* the intended outputs moved.

### Acceptance
- Running it twice on unchanged `master` produces identical baselines (deterministic capture; for the RNG algorithms
  capture with the seed pinned so the "before" is reproducible).
- Manifest records git SHA + platform + numpy/ffmpeg versions.

**PR gate:** script runs clean on the reference platform; captured baselines committed (or documented as generated
artifacts, per repo convention).

---

## CP0.2 — Reference oracles & resource-stability harness
**Branch:** `test/reference-oracles`  ·  Supports Phases 1, 3, 4, 5, 6

### Deliverables
1. **numpy percentile oracle** (`test/remediation/oracles.py`): helper asserting an essentia result matches
   `numpy.percentile(a, q)` over a sweep of array sizes and quantiles, `rtol=1e-6`. Used by Phase 2 CP2.1.
2. **ffmpeg decode oracle**: helper that shells out to `ffmpeg -i <file> -f f32le -` (or `-map` for the target stream),
   parses the raw float stream, and compares against an essentia `AudioLoader` decode — tolerance 0 for lossless,
   1e-6 for lossy, and an **exact sample-count** assertion. Used by Phase 3 CP3.1/CP3.2. Skip cleanly (not fail) if
   `ffmpeg` is not on PATH, so lanes without it don't red-flag.
3. **Crafted-file generator**: helper producing a FLAC with maximal block size / high channel count for the SEC-01
   overflow test (via ffmpeg CLI), plus a non-audio `.txt` for the leak tests.
4. **Resource-stability harness** (psutil): context manager that samples RSS and open-FD/handle count, runs a callable
   N times, and asserts growth stays below a threshold. Used by every leak/RSS test in Phases 1, 4, 5. Provide both a
   POSIX FD count and a Windows handle count path.

### Acceptance
- Each oracle is self-testing: feed it a known-good and a known-bad input and confirm pass/fail respectively (an oracle
  that can't fail is rejected in review).
- The psutil harness detects a deliberately-leaking dummy and passes a clean dummy.

**PR gate:** oracles + harness importable from the test tree; self-tests green; graceful skip when `ffmpeg`/`psutil`
absent.

---

## CP0.3 — CI matrix confirmation
**Branch:** `ci/remediation-matrix`  ·  Supports Phases 4, 6

### Deliverable
Confirm (and, if missing, add config for) the build/test lanes the later phases require:
- **MSVC + gcc + clang** full-suite lanes (Phase 6 exercises all three `RogueVector` `#ifdef` branches and the
  threading refactor). Include a **clang + ThreadSanitizer** and a **clang + AddressSanitizer** lane for the streaming
  tests.
- **NumPy 1.x and NumPy 2.x** Python-binding lanes (Phase 4 / DESIGN-04 must build against both).
- Document the exact commands each phase's "PR gate" will invoke, so gates are copy-pasteable, not hand-wavy.

### Acceptance
- A no-op PR runs green on every lane, establishing the baseline matrix.
- The `NPY_NO_DEPRECATED_API` build (Phase 4) is confirmed *not* yet set (so Phase 4's introduction of it is a clean,
  observable change).

**PR gate:** matrix documented in `docs/remediation/ci-matrix.md`; all lanes green on a no-op change.

---

## Phase 0 exit criteria
- Baseline capture reproducible; oracles self-tested and failure-capable; resource harness proven on dummies.
- CI matrix (MSVC/gcc/clang + ASan/TSan + numpy 1.x/2.x) green on a no-op and documented.
- Every later phase's "PR gate" line now names a command that actually exists.
