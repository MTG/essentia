# Phase 1 — Implementation Spec (Batch A)

**Objective:** land the output-preserving one-liners, guards, and leak fixes. Every change here alters behavior
**only for invalid input** (empty vectors, malformed files, error paths) or is a pure perf/leak fix — no valid-output
baseline moves. This is the Opus plan-session artifact for Phase 1; code and review follow it verbatim.

**Phase models:** Code **Sonnet 5** / Review **Opus 4.8**.
**Branch base:** `master`. Each checkpoint is one PR off `master`, mergeable in any order (no inter-CP dependencies).
**Line numbers** below were verified against the working tree at plan time; re-confirm before editing (they drift).

Global acceptance gate for every CP:
```
python test/src/unittests/all_tests.py io base standard rhythm highlevel machinelearning audioproblems
```
must stay green, plus the CP-specific test named below. Add the new tests in the same PR.

---

## CP1.1 — essentiamath.h correctness + perf cluster
**Branch:** `fix/essentiamath-guards`  ·  **Findings:** BUG-03, BUG-18, BUG-17, PERF-02
**File:** `src/essentia/essentiamath.h`

### BUG-03 — Pearson zero-variance guard ignores `yStddev` (line 1243)
Current:
```cpp
if ((xStddev == (T)0.0) || (xStddev == (T)0.0) || (xStddev == (T)0.0)) return (T) 0.0;
```
Target:
```cpp
if ((xStddev == (T)0.0) || (yStddev == (T)0.0)) return (T) 0.0;
```
Why: constant `y` (e.g. DC/silent right channel in `FalseStereoDetector`) currently yields `cov/(x*0)` → Inf/NaN,
which `std::max(std::min(NaN,1),-1)` propagates. The return-0 convention is already documented one line above.

### BUG-18 — `transpose` hard-codes `Real` for the row type (line 1063)
Current:
```cpp
std::vector<std::vector<T> > result(ncols, std::vector<Real>(nrows));
```
Target:
```cpp
std::vector<std::vector<T> > result(ncols, std::vector<T>(nrows));
```
Why: the inner `std::vector<Real>` breaks (or silently converts) any `transpose<T>` where `T != Real`.
Behavior-identical for the existing `Real` callers.

### BUG-17 — `hist` reads `cutoff[0]` when `n_bins == 1` (lines 982, 992, 1007)
`cutoff` has `n_bins - 1` elements; with `n_bins == 1` line 1007 (`T current_cutoff = cutoff[0]`) reads element 0 of
an empty vector (UB), and `n_bins == 0` underflows the loop bounds. Add an early-out at the top of `hist` (after the
min/max lines) before `cutoff` is built:
```cpp
if (n_bins == 0)
  throw EssentiaException("hist: n_bins must be >= 1");
if (n_bins == 1) {
  x_array[0] = (miny + maxy) / (T)2.0;
  n_array[0] = (int)n;
  return;
}
```
Why: single-bin histogram = all `n` samples in bin 0, center at the midpoint. No valid multi-bin path changes.

### PERF-02 — `sumSquare` takes its vector by value (line 106)
Current:
```cpp
template <typename T> T sumSquare(const std::vector<T> array) {
```
Target:
```cpp
template <typename T> T sumSquare(const std::vector<T>& array) {
```
Why: eliminates a full copy per call (ConstantQ configure path today; cheap insurance for any future hot caller).

### Tests
- `test/src/unittests/audioproblems/test_falsestereodetector.py`: stereo input, constant right channel →
  correlation output **exactly 0** (not NaN).
- Add a small direct case exercising a `hist` caller with `n_bins=1` (or a C++ unit if the harness supports it):
  assert all counts land in bin 0, center == midpoint.
- BUG-18/PERF-02: covered by a clean build + existing suite unchanged.

**PR gate:** build + `audioproblems` + `base` suites green; falsestereo NaN case added.

---

## CP1.2 — yaml/json robustness & leaks
**Branch:** `fix/yamlinput-json-robustness`  ·  **Findings:** SEC-03, BUG-12, BUG-14

### SEC-03 — unchecked `ftell` + leaked `new char[]` (`src/algorithms/io/yamlinput.cpp:100-112`)
Current uses `size_t filesize = ftell(file)` then `char* jsonChar = new char[filesize]`, freed only on the success
path — `JsonConvert(...).parseDict()` throws routinely on malformed JSON, leaking the buffer each time; and a
non-seekable stream makes `ftell` return `-1` → `SIZE_MAX` allocation.
Target: validate the offset and use RAII.
```cpp
fseek(file, 0, SEEK_END);
long ftellResult = ftell(file);
if (ftellResult < 0)
  throw EssentiaException("YamlInput: could not determine json file size");
size_t filesize = (size_t)ftellResult;
rewind(file);
std::vector<char> jsonChar(filesize);
size_t result = fread(jsonChar.data(), sizeof(char), filesize, file);
if (result != filesize)
  throw EssentiaException("YamlInput: error reading the json file");
string yamlString = JsonConvert(string(jsonChar.data(), filesize)).parseDict();
```
(Drop the manual `delete[] jsonChar;`.)

### BUG-12 — YAML AST leaked on non-mapping root / `updatePool` throw (`yamlinput.cpp:134-144`)
Current `YamlNode* root` / `YamlMappingNode* rootMap` are `delete`d only on the happy path. Wrap `root` in a
`std::unique_ptr<YamlNode>` right after parsing and drop the manual `delete rootMap;`:
```cpp
std::unique_ptr<YamlNode> root(...);            // from the parse result
YamlMappingNode* rootMap = dynamic_cast<YamlMappingNode*>(root.get());
if (!rootMap) throw EssentiaException("YamlInput: root node is not a mapping node, ...");
updatePool(rootMap, &p, "");                    // may throw; unique_ptr still frees root
```
Note: `root` is currently assigned inside the try/catch — keep the parse there and move ownership into the
`unique_ptr` after the catch, or restructure so the pointer is owned from creation. Verify `parseYaml`'s ownership
contract (caller-owns) before finalizing.

### BUG-14 — `countBackSlashes` wrong sign (`src/essentia/utils/jsonconvert.cpp:43`)
Current:
```cpp
return -_pos - 1 - i;
```
Target:
```cpp
return _pos - 1 - i;
```
Why: today it returns a negative, wrong-magnitude count; the sole caller only tests parity, which survives by
accident (differs by `2*_pos`). Fix now so any future magnitude use is correct. Escaped-quote parsing must be
byte-identical after the fix.

### Tests
- `test/src/unittests/io/test_yamlinput.py`: loop ~1000 malformed-JSON loads each raising `RuntimeError`; assert
  FD/RSS stable (psutil harness from CP0.2). Existing valid-file cases byte-identical.
- Escaped-quote JSON cases (`\"`, `\\"`, `\\\"`) parse identically before/after BUG-14.

**PR gate:** `io` suite green; leak loop stable; escape cases unchanged.

---

## CP1.3 — scheduler / pool / ringbuffer
**Branch:** `fix/scheduler-pool-ringbuffer`  ·  **Findings:** BUG-04, PERF-01, MEM-01

### BUG-04 — null deref in `printNetworkBufferFillState` (`src/essentia/scheduler/network.cpp:969-975`)
Current logs the warning then falls through to `Network::lastCreated->printBufferFillState()`.
Target: add the missing `return`.
```cpp
void printNetworkBufferFillState() {
  if (!Network::lastCreated) {
    E_WARNING("No network created, or last created network has been deleted...");
    return;
  }
  Network::lastCreated->printBufferFillState();
}
```
Reachable from streaming `FrameCutter::process` on the buffer-full path when no network exists (manual acquire/release).

### PERF-01 — `removeNamespace` cleanup (`src/essentia/pool.cpp:96-119`)
Inside the `SEARCH_AND_DESTROY` macro, hoist the prefix once, use a prefix `compare` instead of full-string `find`,
and erase via the returned iterator (drops the fragile `pos`/`tmpIt` dance):
```cpp
MutexLocker lock(mutex##tname);
const string prefix = ns + ".";
map<string, t>::iterator it = _pool##tname.begin();
while (it != _pool##tname.end()) {
  if (it->first.compare(0, prefix.size(), prefix) == 0)
    it = _pool##tname.erase(it);
  else
    ++it;
}
```
Semantics identical (removes keys under `ns.`); `map::erase(it)` return is valid C++11.

### MEM-01 — `RingBufferImpl::reset()` needless realloc (`src/essentia/utils/ringbufferimpl.h:144-151`)
Current `delete[] _buffer; _buffer = new Real[_bufferSize];` at identical size. Drop the pair — the existing buffer is
already `_bufferSize`; just reset the indices/counters:
```cpp
void reset() {
  _writeIndex = 0;
  _readIndex = 0;
  _available = 0;
  _space = _bufferSize;
}
```
(Optional zero-fill only if a downstream consumer relies on cleared contents — current code does **not** clear, so
leave as-is to stay behavior-identical.) Do **not** attempt the `std::vector<Real>` migration here; that belongs to
DESIGN-02 in Phase 6.

### Tests
- Unit driving a streaming FrameCutter with `Network::lastCreated == nullptr` (manual acquire/release) so the
  NO_OUTPUT / buffer-full path runs → assert no crash.
- `test/src/unittests/base/test_pool.py` `removeNamespace` cases unchanged.
- Existing ringbuffer tests green.

**PR gate:** `base` + `standard` suites green; manual-FrameCutter no-crash case added.

---

## CP1.4 — TensorflowPredict guards
**Branch:** `fix/tfpredict-guards`  ·  **Findings:** BUG-15, BUG-10
**File:** `src/algorithms/machinelearning/tensorflowpredict.cpp`

### BUG-10 — empty `outputs` indexes an empty vector (line 112)
Before `if (_outputNames[0] == "")`, guard emptiness:
```cpp
if (_outputNames.empty())
  throw EssentiaException("TensorflowPredict: `outputs` must contain at least one name, "
                          "or [\"\"] to list available nodes");
if (_outputNames[0] == "") { ... }
```

### BUG-15 — unchecked `malloc`/`fread` on graph load (lines 187-188)
Current:
```cpp
const auto data = malloc(fsize);
fread(data, fsize, 1, f);
fclose(f);
```
Target:
```cpp
void* data = malloc(fsize);
if (!data) {
  fclose(f);
  throw EssentiaException("TensorflowPredict: could not allocate memory for the graph file");
}
if (fread(data, fsize, 1, f) != 1) {
  free(data);
  fclose(f);
  throw EssentiaException("TensorflowPredict: could not read the graph file");
}
fclose(f);
```

### Tests
- Python: `TensorflowPredict(graphFilename=<valid>, inputs=[...], outputs=[])` → `RuntimeError` with the clear
  message, no crash.
- Existing `test/src/unittests/machinelearning/test_tensorflowpredict*.py` green (needs the TF-enabled build; if the
  CI lane lacks models, gate behind the usual model-availability skip).

**PR gate:** ML suite green (or skipped consistently with current CI); empty-outputs case added.

---

## CP1.5 — audio I/O error-path leaks
**Branch:** `fix/audio-io-error-path-leaks`  ·  **Findings:** BUG-06, BUG-08, BUG-13

### BUG-06 — `openAudioFile` leaks contexts on throw (`src/algorithms/io/audioloader.cpp:104-156`)
After `avformat_open_input` succeeds, the throws at 104/109/115/120/145/152 leak `_demuxCtx` (and later `_audioCtx` /
`_convertCtxAv`). `closeAudioFile()` is already fully null-safe. Simplest robust form: wrap the post-open body in a
`try { ... } catch (...) { closeAudioFile(); throw; }`, or add `closeAudioFile();` immediately before each throw.
Prefer the try/catch to avoid missing a future throw site.

### BUG-08 — `AudioContext::create` leaks `_muxCtx`/`_codecCtx` on throw (`src/essentia/utils/audiocontext.cpp:54-115`)
After `_muxCtx = avformat_alloc_context()`, throws at 66/71/75/115/125/150/162/171/182 leak `_muxCtx` (and `_codecCtx`
after its alloc), and `_muxCtx` stays non-null so the next `create()` sees corrupt state. Wrap the body in
`try { ... } catch (...) { close(); throw; }` using the existing null-safe `close()`; confirm `close()` nulls the
members it frees.

### BUG-13 — `encodePacket` doesn't restore `frame_size` on the exception path (`src/essentia/utils/audiocontext.cpp:305-...`)
`int tmp_fs = _codecCtx->frame_size;` is set at 307 and restored on the normal path, but the `catch(...)` rethrows
without restoring, corrupting the codec context for the next call. Restore in the catch (or use an RAII restorer):
```cpp
catch (...) {
  _codecCtx->frame_size = tmp_fs;
  // ... existing cleanup ...
  throw;
}
```
An RAII guard (`struct FrameSizeRestorer { AVCodecContext* c; int fs; ~FrameSizeRestorer(){ c->frame_size = fs; } }`)
is cleaner and covers all exit paths — reviewer's choice. Do **not** fold in MEM-02's buffer-reuse here; that's Phase 5.

### Tests
- `test/src/unittests/io/test_audioloader.py`: 500 failed loads of a non-audio `.txt` → FD count stable (psutil).
- `test/src/unittests/io/test_audiowriter.py`: repeated failing `create` configs (bad bitrate/format) → no FD/memory
  growth; then a failing-write-followed-by-successful-write case to prove `frame_size` recovered. Existing outputs
  byte-identical.

**PR gate:** `io` suite green; both leak/stability cases added; audiowriter outputs byte-identical.

---

## Phase 1 exit criteria
- All five PRs merged, each with its CP-specific test.
- Full gate suite green on the merge commit.
- No baseline regeneration required anywhere in Phase 1 (if any stored baseline moves, stop — a "preserving" fix
  wasn't; re-scope it into Phase 2).
- Confirm the psutil leak harness (CP0.2) is wired before CP1.2/CP1.4/CP1.5 land.
