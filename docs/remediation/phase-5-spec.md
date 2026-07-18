# Phase 5 — Implementation Spec (Batches E + F): ML wrapper & audio-writer reuse

**Objective:** stop leaking a full TensorFlow session on every SavedModel configure, and stop re-allocating conversion
buffers + an AVFrame on every encode chunk. Both are resource-lifetime fixes with byte-identical functional output.

**Phase models:** Code **Sonnet 5** / Review **Opus 4.8** — contained blast radius, strong byte-identical / RSS
acceptance tests; the reviewer's job is to confirm no double-free and no behavioral drift.

**Branches:** independent — `fix/tf-savedmodel-session-leak`, `fix/audiowriter-buffer-reuse`. Two parallel PRs.

**Prerequisite:** Phase 0 CP0.2 (psutil RSS harness).

**Verified facts (re-confirm before editing):**
- `tensorflowpredict.cpp:155-157` calls `TF_LoadSessionFromSavedModel(...)` and **discards its return value** (the live
  `TF_Session*`); `reset()` later creates a separate session. Leak per SavedModel configure.
- `audiocontext.cpp:305-429` (`encodePacket`) runs `av_samples_alloc` (321) + `av_frame_alloc` (347) +
  `av_frame_get_buffer` (357) + frees (`av_frame_free`/`av_freep`, 422-423) on **every** chunk, at a size known at
  `create()` time. It also restores `_codecCtx->frame_size` (line 424) — the catch path's missing restore is BUG-13,
  fixed in Phase 1 CP1.5; **coordinate so this phase doesn't collide with that edit.**

Global gate: `io` + `machinelearning` suites green, plus each CP's RSS/byte-identical check.

---

## CP5.1 — BUG-09: capture/close the SavedModel session
**Findings:** BUG-09  ·  **File:** `src/algorithms/machinelearning/tensorflowpredict.cpp:146-166` (openGraph, SavedModel branch)

### The defect (verified)
```cpp
TF_LoadSessionFromSavedModel(_sessionOptions, _runOptions,
  _savedModel.c_str(), &tags_c[0], (int)tags_c.size(),
  _graph, NULL, _status);        // returns a TF_Session* — dropped on the floor
```
The function **returns** a live session (it also populates `_graph`, which is all the code wanted). The dropped session
holds CPU/GPU resources + threads and is never closed; `reset()` then builds a *second* session. Every reconfigure with
a SavedModel leaks a full session — unbounded native growth under model sweeps.

### Fix — two acceptable shapes; pick one and document
**Option A (preferred): use the returned session as `_session`, skip the redundant `reset()` re-creation for this path.**
```cpp
TF_Session* s = TF_LoadSessionFromSavedModel(_sessionOptions, _runOptions,
  _savedModel.c_str(), &tags_c[0], (int)tags_c.size(),
  _graph, NULL, _status);
if (TF_GetCode(_status) != TF_OK)
  throw EssentiaException("TensorflowPredict: Error importing SavedModel ... ", TF_Message(_status));
// adopt the session created for us; ensure any pre-existing _session is closed+deleted first
closeSession();          // whatever the class uses to TF_CloseSession/TF_DeleteSession safely (null-safe)
_session = s;
```
Then ensure the configure flow does **not** also create a fresh session via `reset()` for the SavedModel path (trace
`reset()` and `_isConfigured`/`_session` handling around lines 104-116 — the frozen-graph path still needs its own
session creation; only the SavedModel path should adopt `s`).

**Option B (minimal): close the returned session immediately, keep the existing `reset()` re-creation.**
```cpp
TF_Session* s = TF_LoadSessionFromSavedModel(...);
if (TF_GetCode(_status) != TF_OK) throw ...;
TF_CloseSession(s, _status);
TF_DeleteSession(s, _status);   // graph is already populated; reset() makes the real session
```
Option A is more correct (no wasted second session); Option B is the smaller diff. Reviewer chooses based on how
`reset()` and `_session` are structured — but the leak must be gone either way.

### Validation
- psutil RSS harness (CP0.2): configure a SavedModel-based algorithm 50× in a loop; assert RSS **stabilizes** (bounded
  growth, not linear per-iteration).
- Predictions equal the single-configure result **exactly** for a SavedModel fixture
  (`test/src/unittests/machinelearning/test_tensorflowpredict*.py`) — gated behind the usual model-availability skip.

**PR gate:** 50× configure RSS-bounded; predictions bit-identical to single configure; ML suite green (or consistently
skipped where models absent).

---

## CP5.2 — MEM-02: reuse encode buffers + AVFrame across chunks
**Findings:** MEM-02  ·  **File:** `src/essentia/utils/audiocontext.cpp` (`create()` + `encodePacket()` 305-429)

### The defect (verified)
`encodePacket` allocates the swr output buffer (`av_samples_alloc`), an `AVFrame` (`av_frame_alloc` +
`av_frame_get_buffer`), memcpys the converted samples in, then frees all of it — **every chunk** (thousands per file),
all at a size bounded by `_codecCtx->frame_size` known at `create()`.

### Fix
Allocate once at `create()` time, reuse per chunk, free at `close()`:
- In `create()` (after the codec context is open and `frame_size` is known): `av_samples_alloc` the conversion buffer
  at `frame_size` capacity; `av_frame_alloc` + set `nb_samples=frame_size`, `format`, `ch_layout` + `av_frame_get_buffer`
  once. Store as members (`_encFrame`, `_encBuf`, `_encLinesize`).
- In `encodePacket()`: `av_frame_make_writable(_encFrame)` per iteration; set `_encFrame->nb_samples = size` for the
  chunk (short final chunk must set the *actual* size, not `frame_size`); run swr directly into `_encFrame->data`
  where possible to drop the intermediate memcpy, else swr into `_encBuf` then memcpy as today.
- In `close()`: `av_frame_free(&_encFrame)`; `av_freep(&_encBuf[0])` (null-safe; guard for the never-created case).
- **Keep the short-final-chunk invariant:** `_encFrame->nb_samples = size` for the final partial chunk; PTS math
  (`_pts += frame->nb_samples`, line 408) must use the per-chunk value. Do **not** reintroduce the `frame_size`
  mutation hack — carry the chunk size via `nb_samples` (this also aligns with BUG-13's longer-term note).

### Coordination
- BUG-13 (Phase 1 CP1.5) touches the same `catch(...)` block to restore `frame_size`. If CP5.2 removes the `frame_size`
  mutation entirely (carrying size via `nb_samples`), BUG-13 becomes moot for the new code — sequence so the two don't
  conflict: either land CP1.5 first and let CP5.2 supersede it, or note in CP5.2's PR that it subsumes BUG-13. Confirm
  with whoever owns Phase 1.

### Validation
- `test/src/unittests/io/test_audiowriter.py`: all lossless outputs **byte-identical** to pre-change (WAV/FLAC
  round-trip).
- Short-final-chunk case explicitly tested (a file whose length isn't a multiple of `frame_size`) → correct sample
  count, no trailing garbage/silence.
- Optional: RSS/alloc-count instrumentation showing per-chunk allocations dropped to ~0 on the encode loop.

**PR gate:** audiowriter lossless byte-identical; short-final-chunk correct; no new leaks across repeated writes (psutil).

---

## Phase 5 exit criteria
- SavedModel configure no longer leaks a session; 50× loop RSS-bounded; predictions bit-identical.
- Audio encode reuses one buffer + frame; lossless output byte-identical; short-final-chunk correct; per-chunk
  allocations eliminated.
- `io` + `machinelearning` suites green.

## Handoff notes to the reviewer (Opus 4.8)
1. CP5.1: confirm exactly one live `_session` after configure on **both** the SavedModel and frozen-graph paths — no
   leak, no double-close/double-delete on reconfigure or destruction.
2. CP5.2: confirm the reused `AVFrame` is `av_frame_make_writable` before each fill, the final partial chunk sets
   `nb_samples` correctly, and PTS accounting is unchanged.
3. Confirm the BUG-13 / CP5.2 interaction was coordinated — the `frame_size` restore is either preserved or made moot,
   never left half-applied.
