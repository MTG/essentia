# Phase 4 — Implementation Spec (Batch D): Python binding hardening

**Objective:** stop the Python bindings from silently reading the wrong (or out-of-bounds) memory for non-contiguous /
mis-typed numpy arrays, and make the C-API forward-compatible with NumPy 2.x.

**Phase models:** Code **Opus 4.8** / Review **Opus 4.8** — cross-version NumPy ABI subtlety; symmetric strength is
adequate (no threading/UB-of-last-resort surface as in Phase 6).

**Ordering (matters):** land **CP4.1 (Python-side normalization)** before **CP4.2 (C++-side guard)** so the Python
layer stops feeding non-contiguous arrays before the C++ guard starts rejecting them — otherwise CP4.2 alone would turn
currently-"working" (silently-wrong) user calls into hard errors with no soft path. CP4.3 and CP4.4 are independent.

**Branches:** `fix/py-dtype-contiguity`, `fix/pytypes-contiguity-guard`, `fix/numpy2-accessors`, `fix/parsing-mapvec-leak`.

**Prerequisite:** Phase 0 CP0.3 (NumPy 1.x + 2.x lanes) + CP0.2 (psutil RSS harness for CP4.4).

**Verified facts (re-confirm before editing):**
- `standard.py:85-92` converts dtype and checks contiguity on **different variables** (see CP4.1).
- `pytypes/vectorreal.cpp:46-62` `fromPythonRef` wraps `PyArray_DATA`/`PyArray_SIZE` with **no contiguity/stride
  check**, and uses raw struct access `array->descr->type_num` (line 54) and `array->nd` (line 57).
- `parsing.cpp:53` (`MAP_VECTOR_STRING`) leaks the vector from `VectorString::fromPythonCopy`; the sibling
  `MAP_VECTOR_REAL` (line 42) correctly `delete`s.

Global gate: full Python suite green on **both** NumPy 1.x and 2.x, plus each CP's specific test.

---

## CP4.1 — BUG-11: dtype + contiguity normalization in `standard.py`
**Findings:** BUG-11  ·  **File:** `src/python/essentia/standard.py:85-92`

### The defect (verified)
```python
if type(args[i]).__module__ == 'numpy':
    if arg.dtype == 'float64':
        arg = arg.astype('float32')              # converts into `arg`
        essentia.INFO('...truncated into "single"...')
    if not args[i].flags['C_CONTIGUOUS']:        # checks the ORIGINAL args[i]
        arg = copy(args[i])                      # OVERWRITES arg with a float64 copy
```
For an input that is **both** float64 and non-C-contiguous, the float32 conversion is discarded by the second `copy`.
Also `copy.copy` of a transposed 2-D array yields an F-order array — still not C-contiguous — so it doesn't give the
C++ layer what it needs.

### Fix
Single normalization that handles dtype and layout together, operating on `arg` thereafter:
```python
if type(args[i]).__module__ == 'numpy':
    if args[i].dtype == numpy.float64:
        essentia.INFO('Warning: essentia can currently only accept numpy arrays of dtype '
                      '"single". "%s" dtype is double. Precision will be automatically '
                      'truncated into "single".' % (inputNames[i]))
        arg = numpy.ascontiguousarray(args[i], dtype=numpy.float32)
    else:
        arg = numpy.ascontiguousarray(args[i])   # forces C-order regardless of source layout
```
`numpy.ascontiguousarray` guarantees C-contiguity **and** applies the dtype in one pass; a transposed/F-order input
becomes a genuine C-contiguous buffer. All later references must use `arg` (they already do at line 94+).

### Validation
- `numpy.arange(10, dtype='float64')[::2]` → a simple algo (`Mean`): correct value **and** the float32 path taken (no
  double-precision silently retained).
- A transposed 2-D float32 array into a matrix-input algorithm → correct result (C-contiguous reached the C++ side).
- Existing Python suite unchanged for already-contiguous float32 inputs (the common path).

**PR gate:** both numpy cases pass; full Python suite green; no perf regression on the contiguous-float32 fast path
(ascontiguousarray is a no-op copy there).

---

## CP4.2 — SEC-02: contiguity guard in every `fromPythonRef`
**Findings:** SEC-02  ·  **Files:** all `src/python/pytypes/vector*.cpp` (and any `matrix*`/`tensor*` analogues)
**Depends on:** CP4.1 merged.

### The defect (verified)
`fromPythonRef` wraps `PyArray_DATA(array)` in a `RogueVector` with no C-contiguity/stride check. For a strided view
(`a[::2]`) the C++ reads the wrong samples; for a **negative-stride** view (`a[::-1]`) `PyArray_DATA` points at the
*last* element and the C++ reads `size` elements *forward* — an out-of-bounds read past the base buffer. Reachable via
`streaming.py`, direct `_essentia` calls, and the parameter path (`parsing.cpp`) — not all of which go through CP4.1's
Python normalization.

### Fix
In **every** `fromPythonRef` across the pytypes, after the existing dtype/ndim checks, reject non-contiguous/unaligned
arrays with a clear exception. Defense belongs in C++ regardless of the Python-side fix:
```cpp
if (!PyArray_ISCARRAY_RO(array)) {
  throw EssentiaException("VectorReal::fromPythonRef: expected a C-contiguous, aligned array; "
                          "pass numpy.ascontiguousarray(x) (a slice like x[::2] or a transpose is not contiguous)");
}
```
Apply the identical guard (with the type name adjusted) to `vectorstring.cpp`, `vectorcomplex.cpp`, the matrix/tensor
pytypes, and anywhere else `fromPythonRef` wraps `PyArray_DATA`. Grep for `PyArray_DATA` across `src/python/pytypes` to
enumerate every site.

### Validation
- New `test/src/unittests/python/test_bindings_contiguity.py`: pass `a[::2]`, `a[::-1]`, and a 2-D `a.T` to e.g.
  `Mean` / `FrameCutter`; assert a **clean exception** — not wrong numbers, not a crash (run under ASan to prove the
  `a[::-1]` OOB read is gone).
- ⚠ Behavior change: callers that previously got silently-wrong results now get an exception. Because CP4.1 landed
  first, the *documented* Python entry points already normalize; this guard catches the paths that bypass them.
  Document the change in release notes.

**PR gate:** contiguity test raises cleanly for all three view types; ASan clean on the negative-stride case; full
Python suite green (CP4.1 ensures normal calls still pass).

---

## CP4.3 — DESIGN-04: NumPy 2.x accessor macros
**Findings:** DESIGN-04  ·  **Files:** all `src/python/pytypes/*.cpp` using raw array struct access

### The defect (verified)
`array->descr->type_num` (vectorreal.cpp:54) and `array->nd` (line 57) are pre-1.7 NumPy internals, removed from the
public ABI in NumPy 2.x — the bindings fail to build (or misbehave) against modern NumPy.

### Fix
Replace raw struct access with the public accessor macros across all pytypes:
- `array->descr->type_num` → `PyArray_TYPE(array)`
- `array->nd`             → `PyArray_NDIM(array)`
- any `array->descr`      → `PyArray_DESCR(array)`
- any `array->dimensions` → `PyArray_DIMS(array)`
Then define, in the binding build (once, before the numpy headers):
```c
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
```
so the deprecated paths can't silently creep back — the build fails loudly if any remain.

Grep the whole `src/python` tree for `->descr`, `->nd`, `->dimensions`, `->data` on `PyArrayObject*` to find every
site.

### Validation
- Builds clean against **NumPy 1.x and 2.x** (Phase 0 CP0.3 lanes) with `NPY_NO_DEPRECATED_API` set — a remaining
  deprecated access is a hard compile error, proving completeness.
- Full Python suite green on both NumPy versions (runtime behavior identical).

**PR gate:** both numpy lanes build and pass with the deprecation lock in place.

---

## CP4.4 — BUG-16: `MAP_VECTOR_STRING` leak in `parsing.cpp`
**Findings:** BUG-16  ·  **File:** `src/python/parsing.cpp:47-56`

### The defect (verified)
```cpp
mapVecString[skey] = *((vector<string>*)VectorString::fromPythonCopy(value));   // heap vector never freed
```
`fromPythonCopy` returns a heap-allocated vector that is copied into the map and leaked — one `vector<string>` per dict
key per configure. The sibling `MAP_VECTOR_REAL` branch (line 40-42) does it correctly with `delete rv`.

### Fix
Mirror the REAL branch — own the returned pointer and free it:
```cpp
std::unique_ptr<vector<string> > vs((vector<string>*)VectorString::fromPythonCopy(value));
mapVecString[skey] = *vs;
```
(Or the explicit `rv`/`delete rv` form matching MAP_VECTOR_REAL — `unique_ptr` is exception-safe and preferred.)

### Validation
- psutil RSS harness (CP0.2): a Python loop configuring an algorithm with a `map<string,vector<string>>` parameter
  10⁴× → RSS stable.
- Existing parameter tests unchanged.

**PR gate:** RSS-stable loop passes; parameter suite green.

---

## Phase 4 exit criteria
- Python normalization (CP4.1) handles dtype+layout in one pass; C++ guard (CP4.2) rejects non-contiguous/negative-
  stride arrays cleanly (ASan-proven no OOB).
- All pytypes use NumPy public accessors with `NPY_NO_DEPRECATED_API`; build green on numpy 1.x and 2.x.
- `MAP_VECTOR_STRING` leak closed; RSS stable.
- Full Python suite green on both numpy versions.

## Handoff notes to the reviewer (Opus 4.8)
1. Confirm CP4.1's `ascontiguousarray` is a no-op copy on the common contiguous-float32 path (no perf regression) and
   genuinely materializes C-order for the transposed-2D case.
2. Enumerate every `fromPythonRef` / `PyArray_DATA` site and confirm CP4.2's guard is on **all** of them, not just
   `vectorreal`.
3. Confirm the ordering was respected (CP4.1 before CP4.2) so normal user calls don't regress to exceptions.
4. Verify `NPY_NO_DEPRECATED_API` is defined before any numpy header include in the build so it actually bites.
