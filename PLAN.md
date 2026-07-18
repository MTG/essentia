# Essentia Fable-Audit Remediation — Phased Execution Plan

Source: Fable high-effort audit (32 findings). Every phase runs the same loop:
**(1) Opus plan session** → **(2) code** → **(3) code-review** → iterate to green → **(4) merge checkpoint PRs**.
Each checkpoint below = one self-contained, independently revertible PR.

Per-phase implementation specs live in `docs/remediation/phase-N-spec.md` and are produced by that phase's
Opus plan session before any code is written.

---

## Model recommendation cheat-sheet

| Work character | Code model | Review model | Why |
|---|---|---|---|
| Mechanical one-liners, guards, leak-plugs (output-preserving) | Sonnet 5 | Opus 4.8 | Low ambiguity; strong reviewer catches the occasional wrong-variable/edge slip |
| Numerical correctness, output-changing (baselines move) | Opus 4.8 | Fable 5 (high) | Needs reference-comparison reasoning (numpy/ffmpeg); reviewer must independently re-derive |
| Security-critical / FFmpeg buffer + decode-loop | Opus 4.8 | Fable 5 (high) | Heap-overflow + multi-frame semantics; deepest verification tier |
| Python C-API / NumPy ABI hardening | Opus 4.8 | Opus 4.8 | Cross-version ABI subtlety; symmetric strength adequate |
| Memory/perf reuse refactors | Sonnet 5 | Opus 4.8 | Contained blast radius, byte-identical acceptance test |
| Threading + STL-internals refactors (RogueVector, condvar) | Opus 4.8 | Fable 5 (high) | Undefined-behavior surface; multi-platform; highest review tier |

> Convention: **Code / Review** is written per phase; individual checkpoints inherit the phase pair unless noted.

---

## Phase 0 — Validation scaffolding (no product code)
**Code: Sonnet 5 / Review: Opus 4.8**
Nothing else can be validated safely until this exists. No behavior change; merge first.
Detailed spec: `docs/remediation/phase-0-spec.md`.

- [ ] **CP0.1** — Baseline capture harness: script that runs
  `python test/src/unittests/all_tests.py io base standard rhythm highlevel machinelearning audioproblems`
  and stores pass/fail + numeric baselines for the algorithms touched by Batches B/C/G
  (ChromaCrossSimilarity, CrossSimilarityMatrix, FrameCutter streaming, TempoTapDegara, AudioLoader).
- [ ] **CP0.2** — Reference oracles: helper to compare against `numpy.percentile` (rtol 1e-6) and against
  `ffmpeg -i f -f f32le -` full decodes (tol 0 lossless / 1e-6 lossy), plus a `psutil` RSS/FD-stability harness
  for the leak tests.
- [ ] **CP0.3** — CI matrix note: confirm build path (waf/CMake) on the platforms in scope
  (MSVC + gcc + clang for Phase 6; NumPy 1.x + 2.x for Phase 4).

**Plan-session focus:** lock the exact acceptance gates each later phase will cite so "green" is unambiguous.

---

## Phase 1 — Output-preserving one-liners, guards & leak fixes (Batch A)
**Code: Sonnet 5 / Review: Opus 4.8**
All strictly correctness-preserving of *valid* outputs. Fully parallelizable across the 5 PRs.
Detailed spec: `docs/remediation/phase-1-spec.md`.

- [ ] **CP1.1 — essentiamath.h cluster** — BUG-03 (pearson `yStddev` guard), BUG-18 (`transpose` template typo),
  BUG-17 (`hist` `n_bins==1`/`==0` guard), PERF-02 (`sumSquare` by const-ref).
- [ ] **CP1.2 — yaml/json robustness & leaks** — SEC-03 (`ftell>=0` + `vector<char>`), BUG-12 (`unique_ptr` AST),
  BUG-14 (`countBackSlashes` sign).
- [ ] **CP1.3 — scheduler/pool/ringbuffer** — BUG-04 (`printNetworkBufferFillState` missing `return`),
  PERF-01 (`removeNamespace` rewrite), MEM-01 (drop same-size realloc in `reset()`).
- [ ] **CP1.4 — TensorflowPredict guards** — BUG-15 (`malloc`/`fread` checks), BUG-10 (empty-`outputs` guard).
- [ ] **CP1.5 — audio I/O error-path leaks** — BUG-06 (`openAudioFile` cleanup), BUG-08 (`AudioContext::create`
  cleanup), BUG-13 (restore `frame_size` on exception).

---

## Phase 2 — Output-changing numerical correctness (Batch B)
**Code: Opus 4.8 / Review: Fable 5 (high)**
Each PR *intentionally* moves baselines; keep isolated and regenerate affected fixtures in the same PR.
Detailed spec: `docs/remediation/phase-2-spec.md`.

- [ ] **CP2.1 — `percentile()` (BUG-01)** — numpy-style whole-index case + remove unsafe single-element special case.
  Regenerate ChromaCrossSimilarity / CrossSimilarityMatrix baselines.
- [ ] **CP2.2 — FrameCutter ADD_NOISE offset (BUG-05)** — fix source offset (`inputFrame = frame`).
  Regenerate silent-intro noise-frame baselines.

---

## Phase 3 — AudioLoader decode path (Batch C) — security-critical, sequential
**Code: Opus 4.8 / Review: Fable 5 (high)**
One branch; checkpoints land in order. CP3.1 is the heap-overflow fix — fast-track as its own hotfix PR.
Detailed spec: `docs/remediation/phase-3-spec.md`.

- [ ] **CP3.1 — SEC-01** — clamp `swr_convert` output to `FFMPEG_BUFFER_SIZE/(nChannels*4)`; drain swr FIFO in a loop.
- [ ] **CP3.2 — BUG-02** — multi-frame `avcodec_receive_frame` loop + `EAGAIN` drain/resend.
- [ ] **CP3.3 — BUG-07** — `av_channel_layout_copy` + throw on `nb_channels<=0` + `av_channel_layout_uninit`.
- [ ] **CP3.4 — DESIGN-01** — delete dead `decode_audio_frame`; extract shared `convertFrameToBuffer`.

---

## Phase 4 — Python binding hardening (Batch D)
**Code: Opus 4.8 / Review: Opus 4.8**
Sequence SEC-02 after BUG-11 so the Python layer stops feeding non-contiguous arrays first.
Detailed spec: `docs/remediation/phase-4-spec.md`.

- [ ] **CP4.1 — BUG-11** — single-pass `ascontiguousarray(dtype=float32 if float64)` normalization in `standard.py`.
- [ ] **CP4.2 — SEC-02** — `PyArray_ISCARRAY_RO` guard in *every* `fromPythonRef` (all pytypes).
- [ ] **CP4.3 — DESIGN-04** — `PyArray_TYPE`/`PyArray_NDIM`/`PyArray_DESCR` accessors + `NPY_NO_DEPRECATED_API`.
- [ ] **CP4.4 — BUG-16** — `unique_ptr<vector<string>>` in `parsing.cpp` MAP_VECTOR_STRING path.

---

## Phase 5 — ML wrapper + audio writer reuse (Batches E + F)
**Code: Sonnet 5 / Review: Opus 4.8**
Independent of each other; two parallel PRs.
Detailed spec: `docs/remediation/phase-5-spec.md`.

- [ ] **CP5.1 — BUG-09** — capture the `TF_LoadSessionFromSavedModel` session (use as `_session` or close+delete).
- [ ] **CP5.2 — MEM-02** — allocate encode buffer + AVFrame once in `create()`, `av_frame_make_writable` per chunk.

---

## Phase 6 — Threading & core-refactor hardening (Batch G)
**Code: Opus 4.8 / Review: Fable 5 (high)**
Highest-UB-surface phase. RogueVector step 3 (span migration) is explicitly **deferred**.
Detailed spec: `docs/remediation/phase-6-spec.md`.

- [ ] **CP6.1 — DESIGN-02** — replace hand-rolled `Condition` with `std::mutex` + `std::condition_variable`.
- [ ] **CP6.2 — DESIGN-06** — `Network::lastCreated` best-effort: guard writes, document diagnostic-only.
- [ ] **CP6.3 — DESIGN-05** — per-instance `std::mt19937` + fixed seed at the four `rand()` sites.
- [ ] **CP6.4 — DESIGN-03 (steps 1–2)** — delete/deep-copy RogueVector copy-ctor & assignment; startup layout check.

---

## Phase 7 — Optional optimization (Batch H)
**Code: Sonnet 5 / Review: Opus 4.8**
Detailed spec: `docs/remediation/phase-7-spec.md`.

- [ ] **CP7.1 — MEM-03** — publish clamped `duration*sampleRate` through the pool; `audio.reserve()` before run.

---

## Cross-phase gate (after every phase)
Build (waf/CMake per platform) + `python test/src/unittests/all_tests.py io base standard rhythm highlevel
machinelearning audioproblems` all green. Phases 2, 3, 6 additionally pass their per-checkpoint reference
comparisons. Every checkpoint is one PR, self-contained and independently revertible.

## Suggested phase ordering & parallelism
- **Serial priority spine:** Phase 0 → CP3.1 (security hotfix, pull forward) → Phase 1 → Phase 2 → rest of Phase 3.
- **Parallelizable once Phase 0 lands:** Phases 1, 4, 5, 7 are mutually independent.
- **Phase 6 last** (depends on CP1.3/BUG-04; highest risk).

## Out of scope (needs a dedicated follow-up pass — Fable's stated blind spots)
tonal/, sfx/, synthesis/, extractors; audiowriter.cpp internals; Emscripten/JS build; Vamp/Gaia integrations;
RogueVector step 3 (span migration). Recommend `/code-review ultra` on those before closing the audit.
