# Phase 7 — Implementation Spec (Batch H): optional output-vector reservation

**Objective:** remove the transient ~1.5× memory peak when loading long files by reserving the output `StereoSample`
vector up front instead of letting it grow by doubling. Pure optimization; output byte-identical.

**Phase models:** Code **Sonnet 5** / Review **Opus 4.8** — small, contained, byte-identical acceptance test.

**Branch:** `perf/audioloader-reserve-output`. Single checkpoint. Optional — schedule only after the correctness phases
land; nothing depends on it.

**Prerequisite:** Phase 3 merged (the decode path is final) so the reserve doesn't interact with an in-flux loader.

**Verified facts (re-confirm before editing):**
- `src/algorithms/io/audioloader.cpp:590-593` carries the in-code FIXME:
  ```cpp
  // FIXME:
  // _audio.reserve(sth_meaningful);
  _network->run();
  ```
- A 1-hour 44.1 kHz stereo file is ~1.2 GB of `StereoSample`; vector doubling gives ~1.5× transient peak + O(log n)
  full copies.
- Sample rate / duration reach `compute()` via the pool (`internal.sampleRate` etc., lines 595-598), but **after**
  `run()`. Duration must come from the demuxer metadata before the run.

---

## CP7.1 — MEM-03: reserve the output vector from estimated duration
**Findings:** MEM-03  ·  **File:** `src/algorithms/io/audioloader.cpp` (standard `compute()` + streaming loader)

### Fix
1. In the **streaming** loader, expose the container duration when the file is opened. `AVFormatContext::duration`
   (in `AV_TIME_BASE` units) gives total seconds; `sampleRate` is already known at open time. Publish an estimate —
   `estimatedSamples = ceil(duration_seconds * sampleRate)` — through the pool (a new `internal.estimatedSamples`
   value), analogous to the existing `internal.sampleRate` push.
2. In the standard `compute()`, before `_network->run()`, reserve when the estimate is sane:
   ```cpp
   Real durSec = _pool.value<Real>("internal.estimatedSamples"); // or read duration + sr
   if (durSec > 0) {
     size_t est = (size_t)std::min<double>(durSec, kReserveCap);  // clamp against bad metadata
     audio.reserve(est);
   }
   _network->run();
   ```
   Ordering caveat: the estimate must be available **before** `run()`. If the current pool values are only populated
   during the run, push the estimate from `openAudioFile`/`reset()` (which runs at configure time) rather than from the
   streaming process loop. Trace `pushChannelsSampleRateInfo` (audioloader.cpp:524) as the model for an early push.
3. **Clamp** the reservation (`kReserveCap`, e.g. a few minutes × 48 kHz, or a byte budget) so corrupt/oversized
   duration metadata can't trigger a pathological `reserve`. Over-reservation on bad metadata is bounded; under-
   estimation just falls back to the current doubling behavior (correct, only slightly slower).

### Validation
- `test/src/unittests/io/test_audioloader.py`: all decodes **byte-identical** (reserve changes capacity, never
  contents or size).
- A file with **missing/zero duration** metadata still decodes correctly (falls back to growth; no crash, no
  truncation).
- A file with **absurd duration** metadata reserves only up to the clamp (no OOM attempt) and still decodes correctly.
- Optional peak-RSS benchmark on a long file showing the transient peak dropped toward ~1× and full-copy reallocations
  eliminated.

**PR gate:** `io` suite byte-identical; zero-duration and absurd-duration files handled safely; optional RSS benchmark
attached.

---

## Phase 7 exit criteria
- Output vector reserved from a clamped duration estimate; long-file transient peak reduced; decodes byte-identical.
- Missing / corrupt duration metadata degrades gracefully to the current growth behavior.
- `io` suite green.

## Handoff notes to the reviewer (Opus 4.8)
1. Confirm the estimate is available **before** `run()` (not read from a pool value only populated during the run).
2. Confirm the clamp makes a maliciously large `duration` harmless (no unbounded `reserve`).
3. Confirm `reserve` never changes the vector's final `size()` — only capacity — so all downstream sizes and outputs
   are byte-identical.
