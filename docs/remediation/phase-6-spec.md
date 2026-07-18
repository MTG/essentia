# Phase 6 — Implementation Spec (Batch G): Threading & core-refactor hardening

**Objective:** remove the undefined-behavior and thread-safety hazards in the streaming core: the hand-rolled
`Condition` primitive (leaks OS handles, single-waiter wake semantics), the mutable-global `Network::lastCreated`,
process-global `rand()` in four algorithms, and `RogueVector`'s unsafe copy-constructor + silent STL-internals poking.

**Phase models:** Code **Opus 4.8** / Review **Fable 5 (high)** — this is the highest-UB-surface phase: memory-layout
assumptions, cross-platform threading, and a real-time hot path. The reviewer must build and reason across MSVC + gcc +
clang and independently verify no deadlock/lost-wakeup was introduced.

**Ordering (important):**
1. **CP6.1 (Condition → std::condition_variable)** lands first — it also completes MEM-01's buffer question and is the
   riskiest to sequence against CP6.4 (both touch the streaming hot path).
2. **CP6.2 (lastCreated)** depends on Phase 1's CP1.3/BUG-04 null-guard already being merged.
3. **CP6.3 (RNGs)** is independent; can run in parallel but is output-changing (rhythm baselines) so keep it isolated.
4. **CP6.4 (RogueVector steps 1–2)** lands last. **Step 3 (span migration) is explicitly out of scope** — do not
   attempt the PhantomBuffer view redesign in this phase.

**Branches:** one per checkpoint (`fix/condition-stdcv`, `fix/network-lastcreated`, `fix/per-instance-rng`,
`fix/roguevector-safety`). CP6.1 and CP6.4 must **not** be in flight simultaneously against the same files.

**Prerequisites:** Phase 0 CP0.1/CP0.3 (baseline capture + the MSVC/gcc/clang build matrix and NumPy 1.x/2.x lanes);
a threaded producer/consumer stress harness (add in CP6.1 if not already present).

**Verified facts (re-confirm before editing — lines drift):**
- `Condition` is defined twice in `src/essentia/utils/ringbufferimpl.h`: Win32 (lines 29-75, auto-reset `event`, **no
  destructor** → leaks the HANDLE + two CRITICAL_SECTIONs) and pthread (82-98, **no destructor** → leaks mutex/cond).
- `RingBufferImpl` uses it via `condition.lock()/wait()/unlock()` in `waitAvailable`/`waitSpace` (153-183) and
  `condition.lock()/signal()/unlock()` in `add`/`get` (203-211, and the symmetric block in `get`). `_available` /
  `_space` are `Atomic`.
- `Network::lastCreated` is a raw `static Network*` (`network.h:203`), set unconditionally in the ctor
  (`network.cpp:173`) and nulled only when the destructed network *is* the last one (`185`).
- Production `rand()` sites: `synthesis/stochasticmodelsynth.cpp:125`, `io/audioonsetsmarker.cpp:70` and `:128`,
  `rhythm/tempotapdegara.cpp:433`. **`NoiseAdder` is the reference implementation** — `std::mt19937 _mtrand;`
  (`noiseadder.h:45`, with an `MTRand` fallback), a `fixSeed` parameter, seeded in `configure()`
  (`noiseadder.cpp:39-41`). (`test/**` `rand()` uses are out of scope.)
- `RogueVector` (`src/essentia/roguevector.h`): copy-ctor (42-45) always builds a **non-owning** alias
  (`_ownsMemory=false`) of the source's `data()`; per-STL `setData`/`setSize` poke private internals
  (`*reinterpret_cast<T**>(this)` for clang/emscripten, `_M_impl._M_start` for libstdc++, `_Myfirst()` for MSVC).

Global acceptance gate: full suite on **each** platform +
```
python test/src/unittests/all_tests.py io base standard rhythm highlevel machinelearning audioproblems
```
green, plus each CP's specific validation.

---

## CP6.1 — DESIGN-02: replace `Condition` with `std::mutex` + `std::condition_variable`
**Findings:** DESIGN-02 (+ folds MEM-01's buffer concern)  ·  **Files:** `src/essentia/utils/ringbufferimpl.h`

### The defect
- **Leaks:** neither `Condition` variant has a destructor — every `RingBufferImpl` (one per RingBufferInput/Output)
  leaks an event HANDLE + 2 CRITICAL_SECTIONs (Win32) or a mutex + cond (pthread) for the life of the process, and on
  every network teardown/rebuild.
- **Lost-wakeup risk:** the Win32 variant uses an **auto-reset** event with a manual `waitersCount` guard — classic
  single-waiter wake semantics; correct only because there is exactly one waiter per buffer today, but fragile and not
  what the pthread variant does. C++11 (already required by this codebase) gives a portable, leak-free primitive.

### Fix
Delete **both** `#ifdef OS_WIN32 … #else … #endif` `Condition` class blocks (lines 25-101) and replace the
`RingBufferImpl` synchronization with standard types:

```cpp
#include <mutex>
#include <condition_variable>

// members (replace `Condition condition;`):
std::mutex _mutex;
std::condition_variable _cond;
```

Rewrite the four interaction points as predicate waits + `notify_all`:

```cpp
void waitAvailable() {
  std::unique_lock<std::mutex> lk(_mutex);
  _cond.wait(lk, [this]{ return _available != 0; });   // predicate guards against spurious/lost wakeups
}

void waitSpace() {
  std::unique_lock<std::mutex> lk(_mutex);
  _cond.wait(lk, [this]{ return _space != 0; });
}

// in add(), after updating _space/_available:
{
  std::lock_guard<std::mutex> lk(_mutex);
  if (_waitingCondition == kAvailable) _cond.notify_all();
}

// in get(), symmetric block: notify when _waitingCondition == kSpace.
```

Implementation notes for the coder:
- `_available` / `_space` are `Atomic`. To make the predicate wait correct, the **producer/consumer must update the
  counter and notify under the same `_mutex`** the waiter uses — otherwise a wakeup can be lost between the predicate
  check and the wait. Simplest correct approach: keep the atomics for the fast-path reads in `add`/`get`, but perform
  the counter change that the predicate observes, and the `notify_all`, inside the locked block. Confirm there is no
  path that increments `_available`/`_space` without notifying while a waiter is parked.
- Use `notify_all` (not `notify_one`) — cheap here (≤1 waiter) and removes any single-waiter assumption.
- **MEM-01 tie-in:** once `Condition` is gone you may also migrate `_buffer` from `Real*`/`new[]` to
  `std::vector<Real>` and make `reset()` just reset indices (completing MEM-01). Keep this **optional and in the same
  PR only if it stays behavior-identical**; if it complicates review, leave `_buffer` as-is and let Phase 1's CP1.3
  MEM-01 fix stand.
- No destructor needed — standard types clean themselves up (the leak is fixed by construction).

### Validation
- Existing ring-buffer unit tests green.
- **Threaded stress harness** (add it here): one producer + one consumer, 10⁷ samples, assert every sample transfers
  in order with no loss and the run completes within a generous timeout (no deadlock / no lost wakeup). Run under
  ThreadSanitizer on the clang lane.
- Streaming FrameCutter / RingBufferInput end-to-end tests unchanged.
- Confirm no HANDLE/FD growth across repeated network create/destroy (psutil / Windows handle count).

**PR gate:** all three platforms build; ring-buffer + streaming tests green; TSan clean; stress harness passes; handle
count stable.

---

## CP6.2 — DESIGN-06: harden `Network::lastCreated`
**Findings:** DESIGN-06  ·  **Depends on:** CP1.3 (BUG-04 null-guard already merged)  ·  **Files:** `network.cpp/.h`

### The defect
`lastCreated` is a mutable global set unconditionally in every `Network` ctor and nulled only if the network being
destroyed happens to be the last one. With two live networks (common under threading), the pointer can refer to the
wrong network or a destroyed one. Its only consumer is the diagnostic `printNetworkBufferFillState()`.

### Fix (minimal — do **not** redesign network lifecycle here)
- Make the diagnostic explicitly best-effort and document it as such (a comment on the declaration in `network.h:203`
  and on `printNetworkBufferFillState`).
- Serialize writes so the pointer is never torn / observed half-updated. Either:
  - guard the ctor/dtor writes and the reader with a single `static std::mutex`, **or**
  - make `lastCreated` `thread_local` so each thread sees the last network *it* created (usually what a per-thread
    debugger wants).
  Prefer `thread_local` if the diagnostic is only ever called from the thread driving the network; otherwise the mutex.
  Reviewer decides based on how `printNetworkBufferFillState` is invoked in practice.
- Keep BUG-04's `if (!lastCreated) { warn; return; }` guard (from Phase 1) — this CP builds on it.

### Validation
- Build + full suite unchanged.
- Thread test: two threads each create/destroy a `Network` concurrently in a loop while a third calls
  `printNetworkBufferFillState()` — assert no crash, no use-after-free (TSan/ASan clean). The diagnostic output being
  approximate is acceptable and documented.

**PR gate:** suite green; concurrent create/destroy test ASan+TSan clean.

---

## CP6.3 — DESIGN-05: per-instance RNGs replacing global `rand()`
**Findings:** DESIGN-05  ·  ⚠ **Output-changing** (rhythm baselines move within tolerance) — keep isolated.

### The defect
`rand()` is process-global and not thread-safe: parallel extractors interleave the stream nondeterministically, and
run-to-run reproducibility depends on global state. Four production sites use it.

### Fix — follow the `NoiseAdder` pattern at each site
For each algorithm add a per-instance generator member and (where a seed matters for reproducibility) a `fixSeed`
parameter, mirroring `noiseadder.h:45` / `noiseadder.cpp:39-41`:

```cpp
// header:
std::mt19937 _mtrand;
// configure(): seed deterministically when requested
if (parameter("fixSeed").toBool()) _mtrand.seed(0);
else                               _mtrand.seed(time(NULL) ^ clock());
```

Per-site replacements:
- **`synthesis/stochasticmodelsynth.cpp:125`**
  `phase = 2*M_PI * Real(rand()/Real(RAND_MAX));`
  → draw a uniform in [0,1) from `_mtrand` (`std::uniform_real_distribution<Real>(0,1)` or
  `Real(_mtrand())/_mtrand.max()`).
- **`io/audioonsetsmarker.cpp:70` and `:128`**
  `(rand()/Real(RAND_MAX) * 2.0 - 1.0) * amplitude` → same expression driven by `_mtrand`. One shared member covers
  both sites in this algorithm.
- **`rhythm/tempotapdegara.cpp:433`**
  `observations[t][i] += 0.0001 * observationsMax * (Real)rand()/RAND_MAX;` — this is a tie-breaking epsilon. Use a
  **fixed seed** so beat tracking is reproducible; a `fixSeed`-style default (seed 0) is appropriate since the jitter
  is only to break ties.

Notes:
- Match each algorithm's existing distribution shape exactly (uniform in the same range); only the *source* changes.
- Add `fixSeed` only where the algorithm's contract benefits (TempoTapDegara definitely; the marker/synth are audible
  output, seed default per project convention). Keep parameter naming/description identical to NoiseAdder's.

### Validation
- ⚠ Output changes: the random *stream* differs, so exact samples/beat positions shift. Validate that
  `test/src/unittests/rhythm/test_tempotapdegara.py` stays **within existing tolerances**, and regenerate any baseline
  that stored exact random-dependent values, documenting the change in the PR.
- Reproducibility test: run each affected algorithm twice in-process with `fixSeed=true` (or the fixed-seed default) →
  **identical** output. Previously impossible with global `rand()`.

**PR gate:** rhythm/synthesis/io suites within tolerance; fixed-seed double-run identical; baselines regenerated with
justification.

---

## CP6.4 — DESIGN-03 (steps 1–2 only): RogueVector safety
**Findings:** DESIGN-03  ·  **Files:** `src/essentia/roguevector.h` (+ consumers verified, not redesigned)
**Step 3 (span migration of PhantomBuffer views) is OUT OF SCOPE — file as a separate future epic.**

### The defect
- **Unsafe copy-ctor (lines 42-45):** always creates a non-owning alias (`_ownsMemory=false`) pointing at the source's
  `data()`. If the source is itself owning and dies first, the copy dangles; if the source is non-owning, you get two
  aliases to memory neither controls. Copies of `RogueVector` are a latent use-after-free.
- **Silent STL-internals poking (lines 59-101):** `setData`/`setSize` write private `std::vector` fields per
  implementation. Any STL layout change, or a debug-iterator / hardened / ASan build, corrupts memory with no
  diagnostic. The code itself calls this "a big hack ... very dangerous".

### Fix — the two safe, high-value steps
**Step 1 — make copies safe.** Either delete the copy operations so accidental copies fail at compile time, or make
them deep-copy when the source owns its memory. Deleting is safest if no code actually copies a `RogueVector` (grep
first); deep-copy is the compatible fallback:
```cpp
// Option A (preferred if unused):
RogueVector(const RogueVector<T>&) = delete;
RogueVector<T>& operator=(const RogueVector<T>&) = delete;

// Option B (if copies exist and must own):
RogueVector(const RogueVector<T>& v) : std::vector<T>(v.begin(), v.end()), _ownsMemory(true) {}
```
Also add an assignment-operator decision consistent with the ctor. **Grep every consumer** (phantombuffer, pytypes,
anywhere constructing/copying `RogueVector`) before choosing A vs B; document the audit in the PR.

**Step 2 — make layout breakage loud, not silent.** Add a startup self-check that constructs a real `std::vector<T>`,
records its `data()`/size, drives the same `setData`/`setSize` pokes on a `RogueVector`, and asserts the observable
`data()`/`size()` round-trip. Run it once (e.g. a static initializer guarded by `#ifndef NDEBUG`, or a dedicated unit
test executed on every platform in CI) so a future STL/layout change fails fast with a clear message instead of
silently corrupting memory:
```cpp
// pseudocode for a unit test (preferred over static init):
RogueVector<float> rv;
float storage[4] = {1,2,3,4};
rv.setData(storage); rv.setSize(4);
ASSERT(rv.data() == storage && rv.size() == 4 && rv[3] == 4.0f);
// destruct without freeing storage (non-owning) — must not double-free.
```

### Validation
- Full build on **MSVC + gcc + clang** (the three `#ifdef` branches all exercised).
- The layout self-check test passes on each platform; deliberately breaking a poke (local experiment) makes it fail —
  proving it actually guards.
- Complete Python unit-test suite + framecutter/ringbuffer streaming tests green (RogueVector is load-bearing for
  zero-copy views).
- ASan build of the streaming tests clean (no new use-after-free from the copy-ctor change).

**PR gate:** three-platform build + suites green; self-check test present and effective; ASan clean; consumer copy
audit documented. Step 3 explicitly deferred in the PR description.

---

## Phase 6 exit criteria
- `Condition` class removed; ring buffer uses `std::mutex`/`std::condition_variable`; TSan-clean stress passes; no
  handle/FD leaks across network churn.
- `lastCreated` writes serialized (mutex or thread_local) and documented best-effort; concurrent two-network test
  crash-free.
- All four production `rand()` sites use per-instance `std::mt19937`; fixed-seed runs reproducible; rhythm baselines
  within tolerance and regenerated where needed.
- RogueVector copy operations safe (deleted or deep-copy) and a cross-platform layout self-check guards the STL pokes;
  step 3 filed separately.
- Full gate suite green on **every** platform for each merge.

## Handoff notes to the reviewer (Fable 5, high)
1. **Lost-wakeup audit (CP6.1):** confirm every write to `_available`/`_space` that a parked waiter depends on happens
   under the same `_mutex` as the `wait` predicate, and is followed by `notify_all`. Trace the `Atomic` fast-path reads
   to ensure none bypasses the notify.
2. **No single-waiter assumption remains** — verify `notify_all` and predicate waits make the code correct for N
   waiters even though today N=1.
3. **CP6.3 output changes are intended** — check each site preserves the original *distribution* (range/shape), only
   the source changed; confirm fixed-seed reproducibility and that tolerances, not exact equality, gate the lossy
   comparisons.
4. **CP6.4 copy audit** — independently grep for `RogueVector` copies; confirm option A (delete) vs B (deep-copy) was
   chosen correctly and no consumer relied on the old aliasing copy semantics.
5. Build and run on MSVC, gcc, and clang — do not accept a single-platform green for this phase.
