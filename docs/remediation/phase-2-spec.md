# Phase 2 — Implementation Spec (Batch B): Output-changing numerical correctness

**Objective:** fix two computations that currently produce *wrong* valid-input output. Both fixes **intentionally move
stored baselines** — the old values are incorrect, so the acceptance test is agreement with an independent reference
(numpy) or a re-derived expectation, not the old baseline.

**Phase models:** Code **Opus 4.8** / Review **Fable 5 (high)** — the reviewer must independently re-derive the correct
numerics and confirm each moved baseline is *more* correct, not merely different.

**Branch:** one per checkpoint (`fix/percentile-interp`, `fix/framecutter-addnoise-offset`). Keep them isolated so each
baseline regeneration is atomically revertible.

**Prerequisite:** Phase 0 CP0.1 (baseline capture) + CP0.2 (numpy oracle).

Global gate:
```
python test/src/unittests/all_tests.py io base standard rhythm highlevel machinelearning audioproblems
```
green **after** regenerating the specifically-affected baselines, plus each CP's reference comparison.

---

## CP2.1 — BUG-01: `percentile()` interpolation + single-element bug
**Findings:** BUG-01 (also an OOB read)  ·  **File:** `src/essentia/essentiamath.h:1176-1198`

### The defect (verified)
```cpp
Real k;
int sortArraySize = sorted_array.size();
if (sortArraySize > 1) k = (sortArraySize - 1) * qpercentile;
else                   k = sortArraySize * qpercentile;          // == 1 * q  -> OOB below
Real d0 = sorted_array[int(std::floor(k))] * (std::ceil(k) - k);
Real d1 = sorted_array[int(std::ceil(k))]  * (k - std::floor(k));
return d0 + d1;
```
Two bugs:
1. **Whole-number `k` returns 0.** When `k` is integral, `ceil(k) == floor(k) == k`, so both weights `(ceil(k)-k)` and
   `(k-floor(k))` are 0 → returns 0 instead of `sorted_array[k]`. E.g. `percentile([1,2,3,4,5], 50)` → `k=2.0` → 0
   instead of 3.
2. **Single-element OOB read.** For `size==1`, `k = 1*q`; any `q>0` (e.g. 50 → `k=0.5`) makes `std::ceil(k)=1`, so
   `sorted_array[1]` reads past the one-element array — UB.

Consumers: `ChromaCrossSimilarity` / `CrossSimilarityMatrix` binarization thresholds silently get 0 whenever
`(n-1)*q/100` is integral.

### Fix (match numpy's linear interpolation)
```cpp
std::vector<T> sorted_array = array;
std::sort(sorted_array.begin(), sorted_array.end());

const Real k = (sorted_array.size() - 1) * (qpercentile / 100.0);  // size==1 -> k==0 -> returns element 0
const int lo = int(std::floor(k));
const int hi = int(std::ceil(k));
if (lo == hi) return sorted_array[lo];                             // whole index (incl. size==1, q edges)
return sorted_array[lo] * (hi - k) + sorted_array[hi] * (k - lo);
```
- Remove the `size > 1` special case entirely — with `(size-1)*q`, a single element gives `k=0` and returns the only
  element for every `q`.
- Keep the empty-array exception (line 1177-1178).
- Note the existing signature already divides `qpercentile /= 100.` at line 1183; fold that into the `k` expression or
  keep it — just don't double-divide. Verify against the current code when editing.

### Validation
- **numpy oracle (CP0.2):** sweep array sizes 1..64 × quantiles {0, 25, 33.3, 50, 75, 100}; assert
  `abs(percentile(a,q) - numpy.percentile(a,q)) <= 1e-6 * max(1,|expected|)`.
- Explicit cases: `percentile([1,2,3,4,5],50)==3`; `percentile([7], q)==7` for q ∈ {0,50,100}.
- ⚠ **Regenerate** `ChromaCrossSimilarity` / `CrossSimilarityMatrix` baselines
  (`test/src/unittests/highlevel/test_chromacrosssimilarity.py` and the crosssimilarity tests). In the PR body, show
  for a representative case that the *new* threshold matches numpy and the *old* was 0 — proving the baseline move is a
  correction.

**PR gate:** numpy oracle passes across the sweep; explicit cases pass; affected highlevel baselines regenerated with
before/after evidence; rest of suite unchanged.

---

## CP2.2 — BUG-05: FrameCutter ADD_NOISE source offset
**Findings:** BUG-05  ·  **File:** `src/algorithms/standard/framecutter.cpp:373-379`

### The defect (verified)
The output `frame` is assembled as `[zeros(zeropadSize)] [audio(acquireSize)] [right zero-pad]` (lines 336-360). In the
silent-frame ADD_NOISE branch:
```cpp
case ADD_NOISE: {
  vector<AudioSample> inputFrame(_frameSize, 0.0);
  fastcopy(&inputFrame[0]+zeropadSize, &frame[0], acquireSize);   // BUG: source is &frame[0]
  _noiseAdder->input("signal").set(inputFrame);
  _noiseAdder->output("signal").set(frame);
  _noiseAdder->compute();
  break;
}
```
`fastcopy` reads `acquireSize` samples starting at `&frame[0]` — i.e. the **leading zero-pad**, not the audio. For
frames with left zero-padding (`startFromZero=false`, first frames), the audio is shifted right by `zeropadSize` and
its tail truncated before noise is added.

### Fix
`frame` already holds the correctly zero-padded, correctly positioned audio. Feed it directly:
```cpp
case ADD_NOISE: {
  vector<AudioSample> inputFrame = frame;      // already [zeros][audio][zeros]
  _noiseAdder->input("signal").set(inputFrame);
  _noiseAdder->output("signal").set(frame);
  _noiseAdder->compute();
  break;
}
```
(Equivalently, fix only the source offset: `fastcopy(&inputFrame[0]+zeropadSize, &frame[0]+zeropadSize, acquireSize)`.
The full-copy form is clearer and provably correct for all padding cases — prefer it. Confirm `NoiseAdder` reading and
writing overlapping/separate buffers behaves identically either way; using a separate `inputFrame` preserves current
in/out separation.)

### Validation
- New case in `test/src/unittests/standard/test_framecutter_streaming.py`: silent input, `startFromZero=false`,
  `silentFrames="noise"`. Assert the noise-added output frames match a standard-mode FrameCutter followed by NoiseAdder
  (same seed) within 1e-4 and show **no positional shift** of the underlying (pre-noise) audio.
- ⚠ Output changes for this exact configuration (that *is* the fix). Regenerate any stored baseline for the
  silent-intro / noise path with before/after evidence that the shift is removed.
- All non-silent and non-noise FrameCutter paths byte-identical.

**PR gate:** shift-free noise-frame test passes; affected baseline regenerated with justification; all other FrameCutter
paths byte-identical.

---

## Phase 2 exit criteria
- `percentile` matches numpy across the sweep; single-element path safe; downstream similarity baselines regenerated as
  corrections.
- FrameCutter ADD_NOISE no longer shifts left-padded audio; affected baseline regenerated.
- No baseline outside the two documented sets moves. If anything else moves, a "targeted" fix leaked — stop and
  re-scope.

## Handoff notes to the reviewer (Fable 5, high)
1. Re-derive `percentile` by hand for an integral-`k` case and a single-element case; confirm the new code equals numpy
   and the old returned 0 / read OOB.
2. Confirm the `qpercentile /= 100` handling isn't applied twice after the rewrite.
3. For BUG-05, verify the fix is correct for **both** left-padded (early) frames and fully-populated middle frames, not
   just the failing case.
