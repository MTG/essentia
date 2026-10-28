# Spike ess-rirg.4 — sourcing a TensorFlow C library for linux/aarch64 and macOS arm64 wheels

Status: complete. Epic: ess-rirg. Related: MTG#1478, MTG#1454, MTG#1446, MTG#1486.

## Question

How should `essentia-tensorflow` wheels obtain a TensorFlow C library for

- **(a) linux/aarch64**, where no builder image exists at all, and
- **(b) macOS arm64**, without the deployment-target coupling that currently forces
  `MACOSX_DEPLOYMENT_TARGET=26.2` and produces a `macosx_26_2_arm64` wheel tag?

Today linux x86_64 links a Bazel-built libtensorflow 2.17 baked into MTG's
`mtgupf/essentia-builds:manylinux2014_x86_64` image (`packaging/debian_3rdparty/build_tensorflow.sh`),
and macOS links the Homebrew `tensorflow`/`libtensorflow` bottle
(`cibuildwheel-tensorflow.toml`, commit 66a890f2). The x86_64 macOS job has a FIXME that
disables its own `test-command` because the wheel it produces cannot be installed on the
runner that produced it.

Candidates evaluated: **A** Bazel-build libtensorflow for aarch64; **B** link the C API
exported by the `tensorflow` pip wheel and declare a runtime dependency instead of
vendoring; **C** keep the Homebrew bottle and document the floor; **D** a prebuilt
libtensorflow from a source with trustworthy provenance.

Essentia's TensorFlow surface is small and stable: `src/algorithms/machinelearning/tensorflowpredict.h`
includes only `<tensorflow/c/c_api.h>`, and the algorithms use **29 `TF_*` C-API functions**
(`TF_NewSession`, `TF_SessionRun`, `TF_GraphImportGraphDef`, `TF_LoadSessionFromSavedModel`,
`TF_AllocateTensor`, …). Any library that exports those 29 symbols and ships the C headers
is a candidate. That framing is what makes B and D viable.

## Method

All work was done on an Apple M-series host (macOS 26.6.2, arm64). Nothing was committed on
this branch except this document.

Scratch branches / worktrees used (delete after the replan reads this):

| branch | worktree | purpose |
| --- | --- | --- |
| `scratch/tf-spike-linux` | `…/scratchpad/wt-linux` | source snapshot rsynced into the linux containers |
| `scratch/tf-spike-macos` | `…/scratchpad/wt-macos` | macOS in-tree builds |

Linux evidence came from Docker (colima, native aarch64 VM, **2 vCPU / 8 GiB RAM**,
`quay.io/pypa/manylinux_2_28_aarch64`). That VM size is well under a CI runner and every
build time below should be read as an upper bound for CI, except candidate A where it is
the binding constraint (noted there).

Scripts (kept under `~/tfspike/work/`, logs under the session scratchpad):

- `linux_b.sh` — install `tensorflow==2.17.0`, check symbol coverage, synthesise a
  `tensorflow.pc` pointing into site-packages, build essentia, build the wheel, run
  `auditwheel show/repair` with and without `--exclude`, then exercise four loader strategies.
- `linux_runtime.sh` — re-run the two loader shims against the repaired wheel.
- `macos_b.sh` / `macos_b2.sh` / `macos_real.sh` — the same sequence on macOS arm64 with
  `delocate` in place of `auditwheel`.
- `macos_pure.sh` — control build in which TensorFlow is the *only* non-system dylib.
- `macos_delocate.sh` — delocate flag matrix.
- `bazel_a.sh` — candidate A costing (bounded Bazel sample).

The functional check throughout is `TensorflowPredictEffnetDiscogs` on the real
`discogs-effnet-bs64-1.pb` model (18 MB, downloaded from essentia.upf.edu), fed a
synthesised 16 kHz signal, asserting the embedding shape and finite values. A synthesised
signal was used instead of `MonoLoader` so the test does not drag FFmpeg into the
measurement; it exercises the identical TensorFlow session path.

## Evidence

### 1. Symbol coverage — all four sources export the full C API essentia needs

`nm -D` / `nm -gU`, matched against the 29 functions essentia calls. The pip wheel's
symbols are version-tagged (`TF_NewSession@@tensorflow`), which is why a naive
`grep ' T TF_NewSession$'` reports zero and must not be trusted.

| library source | platform | 29 symbols | notes |
| --- | --- | --- | --- |
| `tensorflow==2.17.0` pip wheel (`libtensorflow_cc.so.2` + `libtensorflow_framework.so.2`) | linux aarch64 | **29/29** | SONAMEs `libtensorflow_cc.so.2`, `libtensorflow_framework.so.2`; max GLIBC 2.17; max GLIBCXX 3.4.19; ships `include/tensorflow/c/c_api.h` |
| `tensorflow==2.17.0` pip wheel (`libtensorflow_cc.2.dylib` + framework) | macOS arm64 | **29/29** | **minos 12.0**, sdk 13.3, `install_name @rpath/…` |
| `libtensorflow-cpu-darwin-arm64.tar.gz` 2.17.0 (`storage.googleapis.com/tensorflow/versions/2.17.0/`) | macOS arm64 | **29/29** | **minos 12.0**; 439 MB + 38 MB; links only system frameworks. Note the `versions/<v>/` path — see §1a |
| conda-forge `libtensorflow` 2.19.1 | linux aarch64 | **29/29** | SONAME `libtensorflow.so.2`; max GLIBC 2.17 but GLIBCXX 3.4.30 / CXXABI 1.3.13, and NEEDED `libabsl_*.so.2505.0.0`, `libprotobuf.so.31.1.0` |
| Homebrew `libtensorflow` 2.21.0 `arm64_linux` bottle | linux aarch64 | **29/29** | SONAME `libtensorflow.so.2`; NEEDED only glibc/libstdc++/libgcc; max GLIBC **2.27**, GLIBCXX 3.4.22 |

The finding that the pip wheel is now where the C API lives is corroborated independently by
the **ess-rirg.2 seat**, which measured the same layout on TensorFlow 2.17.1: the C API in
`tensorflow/libtensorflow_cc.2.dylib` (577 MB, macOS arm64) and
`tensorflow/libtensorflow_cc.so.2` (1037 MB, manylinux x86_64), a complete
`tensorflow/include/tensorflow/c` header tree, no `tensorflow_core` directory, and
`_pywrap_tensorflow_internal.so` reduced to a 2.4 MB pybind shim. That last point is why the
project's old helper link line resolves nothing.

### 1a. Availability of the official C-library tarballs — the channel moved, then stopped

This one needs care, because a naive probe gives the wrong answer, and the ess-rirg.2 seat
and this one initially disagreed about it. **There are two URL schemes**, and the old one was
retired at 2.16. Ranged `GET` (not `HEAD`) against both:

| version | old `…/libtensorflow/libtensorflow-cpu-<plat>-<v>.tar.gz` — darwin-arm64 | — linux-x86_64 | new `…/versions/<v>/libtensorflow-cpu-<plat>.tar.gz` — darwin-arm64 | — linux-x86_64 |
| --- | --- | --- | --- | --- |
| 2.11.0 | 404 | **206** | 404 | 404 |
| 2.15.0 | 404 | **206** | 404 | 404 |
| 2.16.2 | 404 | 404 | **206** | **206** |
| 2.17.0 | 404 | 404 | **206** | **206** |
| 2.18.0 | 404 | 404 | **206** | **206** |
| 2.19.0 | 404 | 404 | 404 | 404 |

So the correct statement is **not** that the channel died at 2.15.0 and that darwin-arm64
never existed — that is exactly what the old path alone shows, and it is what a probe of that
path reported from the ess-rirg.2 seat. TensorFlow **moved** the tarballs to `versions/<v>/`
at 2.16 and **added** a darwin-arm64 build in the new layout. The channel then genuinely
stops after **2.18.0**, on every platform.

This is not inferred from status codes. The 2.17.0 darwin-arm64 tarball was downloaded in
full (104 MB), unpacked, and its `lib/libtensorflow.2.17.0.dylib` inspected directly:
`LC_BUILD_VERSION minos 12.0`, `install_name @rpath/libtensorflow.2.dylib`, 29/29 C-API
symbols, no non-system dependencies. It exists and it is usable. Anyone re-checking this must
use the `versions/<v>/` form.

There has never been an official linux aarch64 build under either scheme.

### 2. The macOS floor is a Homebrew `libtensorflow` bug, not a fact about TensorFlow

Bottle minimum-OS, measured from `LC_BUILD_VERSION` after pulling each bottle from ghcr.io:

| formula | bottle tag | minos |
| --- | --- | --- |
| `libtensorflow` 2.21.0 | `arm64_tahoe` | 26.4 |
| `libtensorflow` 2.21.0 | **`arm64_sequoia`** | **26.2** |
| `libtensorflow` 2.21.0 | `arm64_sonoma` | 15.2 |
| `taglib` 2.3.1 | `arm64_sequoia` | 15.0 |
| `libsamplerate` 0.2.2 | `arm64_sequoia` | 15.0 |
| `libyaml` 0.2.5 | `arm64_sequoia` | 15.0 |
| `fftw` 3.3.11 | `arm64_sequoia` | 15.0 |

Every other formula's Sequoia bottle targets 15.0, as it should. Only `libtensorflow` ships
a Sequoia bottle built with a **26.2** deployment target — one macOS generation above its own
tag. The CI matrix runs on `macos-15` (`.github/workflows/build-wheels-cibuildwheel.yml`),
so `brew install tensorflow` pours the `arm64_sequoia` bottle and hands the build a 26.2
floor. That is exactly the number commit 66a890f2 had to pin, and it explains why the same
job previously needed 15.4 and 14.2: the value tracks whatever machine Homebrew last built
the bottle on, and it drifts upward on Homebrew's schedule, not ours.

Homebrew offers no supported way to pin an older bottle of a non-versioned formula, and the
older bottle we would want (`arm64_sonoma`, minos 15.2) is still above the 15.0 floor the
plain `essentia` wheel already achieves.

### 3. Candidate B end to end — linux/aarch64

Container: `quay.io/pypa/manylinux_2_28_aarch64`, 2 vCPU. `tensorflow==2.17.0` installed
with pip (1.1 GB in site-packages, 20 s). `tensorflow.pc` pointed `libdir` at a directory of
symlinks to the wheel's `.so.2` files and `includedir` at `site-packages/tensorflow/include`.

| step | result |
| --- | --- |
| `waf configure --with-tensorflow` | ok, `Checking for 'tensorflow' : yes` |
| essentia core + bindings build (`-j2`) | **360 s** |
| `_essentia…so` NEEDED | `libtensorflow_cc.so.2`, `libtensorflow_framework.so.2` (plus libessentia, libsamplerate, libtag, libyaml, libstdc++) |
| `pip wheel` (bindings only) | **26 s**, `essentia_tensorflow-…-cp312-cp312-linux_aarch64.whl`, 4,573,712 B |
| `auditwheel show` | "constrains the platform tag to `manylinux_2_27_aarch64`"; lists `libtensorflow_cc.so.2 with versions {'tensorflow'}` as an external dependency |
| `auditwheel repair` **without** `--exclude` | **FAILS**: `Cannot repair wheel, because required library "libtensorflow_cc.so.2" could not be located` |
| `auditwheel repair --exclude libtensorflow_cc.so.2 --exclude libtensorflow_framework.so.2` | **succeeds in 1 s** → `…-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`, **9,216,250 B (9.2 MB)** |

Runtime resolution of the repaired wheel, installed with `pip install --no-deps` into a venv
that also has `tensorflow==2.17.0`:

| loader strategy | result |
| --- | --- |
| (a) plain `import essentia.standard`, no shim | `ImportError: libtensorflow_cc.so.2: cannot open shared object file` |
| (b) `import tensorflow` before `import essentia` | **works** — `TensorflowPredictEffnetDiscogs` → `(3, 1280) float32` |
| (c) `ctypes.CDLL(..., mode=RTLD_LOCAL)` on the two libs | **works** — `(3, 1280)` |
| (c′) same with `mode=RTLD_GLOBAL` | **SIGSEGV (exit 139)** — do not use `RTLD_GLOBAL` |
| (d) `DT_RUNPATH = $ORIGIN/../tensorflow:$ORIGIN/../essentia_tensorflow.libs` on `_essentia.so` **and** `$ORIGIN/../tensorflow:$ORIGIN` on the vendored `libessentia-*.so` | **works with no import-time shim** — `(3, 1280) float32`, first values `[0.03815, -0.00475, -0.01045, 0.07707, 0.29942]` |
| (e) same wheel, `tensorflow` not installed | clean `ImportError: libtensorflow_cc.so.2: cannot open shared object file` |

(d) is the production mechanism: it needs no code in `essentia/__init__.py`, and the failure
mode in (e) is a legible message rather than a crash. Two details the implementation must
respect: `auditwheel repair` already writes `DT_RPATH` = `$ORIGIN/../essentia_tensorflow.libs`
on the extension and `$ORIGIN` on the vendored `libessentia`, so the TensorFlow directory has
to be **prepended to** those values, not substituted for them — replacing them breaks the
load of `libessentia` itself. And **both** binaries need the entry, because both carry a
`NEEDED` on `libtensorflow_cc.so.2`.

### 4. Candidate B end to end — macOS arm64

Built against the pip wheel's dylibs with
`LINKFLAGS="-Wl,-headerpad_max_install_names -Wl,-rpath,@loader_path/../tensorflow -Wl,-rpath,<tf site-packages>"`.
The second, absolute rpath exists only so `delocate` can *resolve* the dylibs at repair time;
delocate's default `--sanitize-rpaths` then deletes it and keeps the `@loader_path` one. That
was verified in the log: `Sanitize: Deleting rpath '…/site-packages/tensorflow' from …libessentia.dylib`.

| step | result |
| --- | --- |
| essentia build (`-j10`, Homebrew deps + TF from the wheel) | **158 s** |
| `_essentia…so` load commands | `@rpath/libtensorflow_cc.2.dylib`, `@rpath/libtensorflow_framework.2.dylib` |
| in-tree `TensorflowPredictEffnetDiscogs` | **`(9, 1280) float32`**, 0.7 s inference |
| `pip wheel` | 283,946 B raw |
| `delocate-wheel --exclude libtensorflow_cc --exclude libtensorflow_framework` | **succeeds in 2 s**, **4,248,201 B (4.25 MB)**; vendors only `libessentia.dylib`, `libsamplerate`, `libtag`, `libyaml` — **no TensorFlow in the wheel** |
| clean venv with `tensorflow==2.17.0`, install the wheel, run effnet | **`EMBEDDINGS (9, 1280) float32`, inference 0.68 s, first values `[0.03815, -0.00475, -0.01045, 0.07707, 0.29942]`** (bit-comparable to linux) |
| same wheel, no `tensorflow` installed | `ImportError: dlopen(…): Library not loaded: @rpath/libtensorflow_cc.2.dylib … tried '…/essentia/../tensorflow/libtensorflow_cc.2.dylib' (no such file)` |

Two delocate failure modes worth recording, because they will bite whoever implements this:

- `delocate-wheel --exclude …` **alone is not enough**. Exclusion is applied after
  resolution, so without an rpath that resolves the dylibs delocate dies with
  `DelocationError: Could not find all dependencies.` The absolute link-time rpath above is
  what fixes it; `--ignore-missing-dependencies` also gets past it.
- With TensorFlow removed from the picture, delocate then enforces the deployment target
  against the *remaining* Homebrew bottles:
  `libtag.2.3.1.dylib has a minimum target of 26.0 … Set MACOSX_DEPLOYMENT_TARGET=26.0`.
  On this Tahoe host that produced a `macosx_26_0_arm64` tag. On the `macos-15` runner the
  same bottles are `arm64_sequoia` at 15.0, so the tag there is `macosx_15_0_arm64`.

**Control experiment.** Rebuilt with `--lightweight=` so TensorFlow is the only non-system
dylib: build 116 s, `delocate --exclude` succeeded and produced a genuine
**`macosx_12_0_arm64`** wheel of **2,364,250 B**, containing only `_essentia.so` (710 KB) and
`libessentia.dylib` (6.7 MB). That build is not shippable (it compiles out `Resample`, which
`essentia.standard` needs at import) but it proves the floor is entirely a property of the
bottles we link, and that TensorFlow from the pip wheel contributes a 12.0 floor.

For comparison, delocating the same control wheel **with** TensorFlow vendored took 19 s and
produced **126 MB**.

### 5. Size and shape of what ships today

`essentia_tensorflow 2.1b6.dev1438` on PyPI:

| wheel | size | tag |
| --- | --- | --- |
| `cp314 … manylinux2014_x86_64.manylinux_2_17_x86_64` | 291.9 MB | manylinux2014 x86_64 |
| `cp314 … macosx_15_0_x86_64` | 122.1 MB | macosx 15.0 |
| `cp314 … macosx_15_0_arm64` | 98.8 MB | macosx 15.0 |

Inside the linux wheel: `essentia_tensorflow.libs/libtensorflow-5bf37f83.so.2.5.0` is
**601,033,073 B uncompressed** of the 643 MB total. TensorFlow is ~93 % of the payload.
PyPI is already granting this project a per-file limit above the 100 MB default, so wheel
size is a bandwidth and disk question, not an upload blocker. `requires_dist` is currently
`numpy>=1.25, pyyaml, six` — there is no `tensorflow` dependency today.

### 6. Candidate A costing — a bounded Bazel sample

Run on the only aarch64 Linux capacity available: the colima VM, **2 vCPU / 8 GiB RAM**,
inside `manylinux_2_28_aarch64`. Bazelisk 1.25, TensorFlow 2.17.0 source, non-interactive
`./configure` (CPU only, no CUDA/ROCm/clang), then
`bazel build --jobs=2 --local_ram_resources=6000 //tensorflow/tools/lib_package:libtensorflow`
with a **2,400 s** wall budget.

| measurement | value |
| --- | --- |
| toolchain install (JDK, python3.12, bazelisk) | 35 s |
| TF 2.17.0 source fetch + unpack | 11 s |
| `./configure` | 3 s, exit 0 |
| bazel build, sampled | **6,628 of 11,062 actions in 2,400 s** (60 %) |
| action total during the sample | grew 10,255 → 11,062 (still being discovered) |
| OOM | **none** at 8 GiB with `--jobs=2` |
| bazel cache after 40 min | 5.1 GB (`build_tensorflow.sh` notes ~9 GB at completion) |

Linear extrapolation from that rate gives **~67 min** on 2 vCPU. That is a floor, for two
reasons visible in the log: the denominator was still growing, and the sampled work was the
cheap host-tool phase (`Compiling xla/literal.cc [for tool]`) rather than the target-side
TensorFlow kernels and the final link of a ~600 MB library. A realistic 2-vCPU figure is
**1.5–3 h**, and a 4-vCPU `ubuntu-24.04-arm` runner should land comfortably under that.

So the honest reading is that candidate A **is feasible inside GitHub's 6-hour job limit** —
which is a different answer than the epic assumed, and the reason the verdict below rests on
cost/benefit rather than on infeasibility.

### 7. What was not measured

Stated plainly so the replan does not over-read the evidence:

- **macOS x86_64 was not tested.** No Intel Mac was available. The Homebrew bottle analysis
  applies (the `sonoma` x86_64 bottle is the only Intel one Homebrew ships for
  `libtensorflow`, which is why that job's floor was 15.2), and the pip wheel does ship an
  `macosx_10_15_x86_64` build, but candidate B was not executed there.
- **linux x86_64 candidate B was not executed**, only aarch64. There is no reason to expect
  a difference — same wheel layout, same SONAMEs — but it is an inference, not a measurement.
- **Candidate D was inspected, not built against.** The official darwin-arm64 tarball and the
  Homebrew `arm64_linux` bottle were verified for symbols, minimum OS, SONAME and ABI, but no
  essentia build was linked against either. Given both have the same shape as the library
  `build_tensorflow.sh` already produces, the risk is low, but it is untested.
- **FFmpeg and chromaprint were excluded from every build** (`--lightweight=…`), so the
  measured wheel sizes omit those vendored libraries and the measured macOS floor omits their
  bottles. FFmpeg's Sequoia bottle would need checking before promising 15.0; on this Tahoe
  host `ffmpeg@7` measures minos 26.0, consistent with the other bottles tracking their tag.
- **Only `TensorflowPredictEffnetDiscogs` was exercised**, on one model, with a synthesised
  signal. The other TensorFlow algorithms share the same session code path but were not run.
- **The candidate A figure is an extrapolation from a 60 % sample**, not a completed build.

## Verdict

### Candidate A — Bazel-build libtensorflow 2.17 for aarch64 → **NO-GO, but on cost, not feasibility**

This is the one candidate whose verdict came out differently from the epic's assumption, so
it deserves a precise statement: **it works**. `./configure` succeeded unmodified on
aarch64, the build ran clean for 40 minutes with no OOM at 8 GiB, and the extrapolated cost
(1.5–3 h on 2 vCPU, less on a 4-vCPU arm runner) fits inside GitHub's 6-hour job limit.
MTG's existing `packaging/debian_3rdparty/build_tensorflow.sh` would need no changes.

It is still the wrong choice, because it is strictly dominated:

- Candidates B and D obtain the identical library — same TensorFlow 2.17, same C API, same
  SONAME — for **20 s** of `pip install` or a single cached download, against 1.5–3 h of
  compute per build.
- It imports ownership this fork does not otherwise need: a fork-owned builder image
  rebuilt and republished on every TensorFlow bump, a registry to host it, and a 5–9 GB
  Bazel cache to manage.
- Nothing in the epic depends on building TensorFlow ourselves. The C API surface essentia
  uses is 29 stable functions; there is no patch we need that upstream does not ship.

Revisit only if both B and D become unavailable — for instance if TensorFlow stops shipping
`libtensorflow_cc` inside the pip wheel and Homebrew drops the `arm64_linux` bottle.

### Candidate B — link the C API from the `tensorflow` pip wheel → **GO**

Proven end to end on both targets. The C API is fully exported (29/29 on both platforms),
the headers ship in the wheel, `auditwheel` and `delocate` both accept the result once
`--exclude` is paired with a resolvable rpath, and `TensorflowPredictEffnetDiscogs` produces
matching embeddings on linux/aarch64 and macOS arm64 from an installed wheel. It removes
TensorFlow from the wheel entirely: 9.2 MB on linux aarch64 and 4.25 MB on macOS arm64,
against 291.9 MB and 98.8 MB today.

The cost is a real change to the user contract — `essentia-tensorflow` stops being
self-contained — and a coupling to the pip wheel's private layout
(`site-packages/tensorflow/libtensorflow_cc.so.2` is not a documented interface; TensorFlow
could rename or drop it, as it already did with the standalone tarballs after 2.18).

### Candidate C — keep the Homebrew bottle → **NO-GO**

The floor is not ours to set. Homebrew's `libtensorflow` Sequoia bottle carries a 26.2
minimum, one generation above its own tag and 11 major versions above the 15.0 floor the
plain `essentia` wheel achieves with the same runner. There is no supported bottle pinning,
the next Homebrew rebuild can move the number again without warning, and the only older
bottle (`arm64_sonoma`, 15.2) is still worse than 15.0. This is the status quo and it is the
thing the epic exists to remove.

### Candidate D — prebuilt libtensorflow with provenance → **GO on macOS, GO on linux/aarch64 with a caveat**

Two usable sources, both with 29/29 symbol coverage:

- **macOS arm64: `libtensorflow-cpu-darwin-arm64.tar.gz` 2.17.0**, published by TensorFlow
  in the same `storage.googleapis.com/tensorflow/versions/` bucket the project's own linux
  x86_64 reference build corresponds to. **minos 12.0**, links only system frameworks, and
  its layout (`include/`, `lib/libtensorflow.2.dylib`) is the exact shape
  `build_tensorflow.sh` produces, so `generate-pc.sh` and the existing `tensorflow.pc`
  handling apply unchanged. This is the cleanest drop-in replacement for the Homebrew bottle.
  **Caveat: the channel is frozen at 2.18.0** — nothing is published for 2.19+ — so it caps
  the TensorFlow version the wheels can offer. **Use the `versions/<v>/` URL form**; the older
  `libtensorflow/libtensorflow-cpu-<plat>-<v>.tar.gz` path was retired at 2.16 and probing it
  makes this source look non-existent (§1a).
- **linux/aarch64: the Homebrew `libtensorflow` 2.21.0 `arm64_linux` bottle.** Monolithic
  `libtensorflow.so.2` (448 MB), SONAME `libtensorflow.so.2`, NEEDED nothing but
  glibc/libstdc++/libgcc, max GLIBC 2.27. That rules out `manylinux2014` but satisfies
  **`manylinux_2_28_aarch64`**, which is the right target for a new architecture anyway.
  Fetchable in CI from ghcr.io with a plain anonymous-bearer `curl`.
  conda-forge's `libtensorflow` linux-aarch64 (2.18.0, 2.19.1) is the fallback: better glibc
  floor (2.17) but it drags `libabsl_*.so` and `libprotobuf.so` as separate NEEDED
  libraries, which auditwheel would have to locate and co-vendor.

### Overall → **GO**, with candidate B as the primary and candidate D as the fallback

The deployment-target coupling is entirely removable. Both B and D produce a macOS arm64
wheel whose floor is set by the ordinary Homebrew audio bottles (15.0 on the current runner)
rather than by TensorFlow, and both give linux/aarch64 a TensorFlow C library without a
fork-owned Bazel build.

## Recommendation

### Runtime contract for users

Adopt candidate B and make `essentia-tensorflow` depend on TensorFlow rather than vendor it:

- add `tensorflow` to the project's `dependencies` for the `essentia-tensorflow` package
  only, pinned compatibly (`tensorflow>=2.17,<2.18` to start, matching the version essentia
  is built and tested against);
- ship no TensorFlow inside the wheel (`auditwheel repair --exclude libtensorflow_cc.so.2
  --exclude libtensorflow_framework.so.2`; `delocate-wheel --exclude libtensorflow_cc
  --exclude libtensorflow_framework`);
- resolve at load time with `DT_RUNPATH` / `LC_RPATH` `$ORIGIN/../tensorflow` and
  `@loader_path/../tensorflow`, applied to **both** `_essentia*.so` and the vendored
  `libessentia` — no shim in `essentia/__init__.py`;
- do **not** add a `ctypes` preload; if one is ever added it must use `RTLD_LOCAL`, since
  `RTLD_GLOBAL` segfaults.

Users then run `pip install essentia-tensorflow` and get TensorFlow pulled in automatically.
The wheel drops from 291.9 MB to 9.2 MB on linux and from 98.8 MB to 4.25 MB on macOS; total
download is roughly unchanged for a user who did not already have TensorFlow, and much
smaller for one who did. Users who deliberately install a different TensorFlow build (GPU,
`tensorflow-cpu`, a conda TensorFlow) get whatever they installed, which is an improvement
over today's silently-vendored copy.

If the replan judges the vendoring change too disruptive for this release, take candidate D
instead: same wheel shape as today, same self-contained contract, but sourced from the
official darwin-arm64 tarball on macOS and the Homebrew `arm64_linux` bottle on linux —
which fixes the deployment target without changing what users install.

### Minimum macOS

**15.0** on the current `macos-15` runner, identical to the plain `essentia` wheel — set by
the Homebrew `taglib`/`fftw`/`libsamplerate`/`libyaml` bottles, not by TensorFlow. Lower
floors are possible but are a separate problem: they require not linking Homebrew bottles at
all. Do not promise a 12.0 floor on the strength of this spike; 12.0 is what TensorFlow
alone allows, and the control experiment that produced a `macosx_12_0_arm64` wheel had the
audio dependencies compiled out.

### CI cost

- Candidate B adds a `pip install tensorflow` to `before-all` on each platform: **20 s** and
  ~1.1 GB of runner disk on linux; comparable on macOS. It removes `brew install tensorflow`
  (a 209 MB bottle) from the macOS jobs.
- `auditwheel repair` drops from vendoring a 601 MB library to **1 s**; `delocate` to **2 s**
  from 19 s.
- Candidate D costs one 104 MB (macOS) or 216 MB (linux) download per job, cacheable.
- Candidate A would cost an estimated 1.5–3 h of aarch64 compute per TensorFlow bump plus a
  fork-owned builder image and a 5–9 GB Bazel cache; both alternatives cost zero build
  minutes.

### Expected wheel matrix

| platform | tag | source of libtensorflow | approx. wheel size |
| --- | --- | --- | --- |
| linux x86_64 | `manylinux_2_28_x86_64` (from `manylinux2014` today) | pip `tensorflow` | ~9 MB |
| linux aarch64 | `manylinux_2_28_aarch64` | pip `tensorflow` | **9.2 MB (measured)** |
| macOS arm64 | `macosx_15_0_arm64` | pip `tensorflow` | **4.25 MB (measured, tagged 26.0 on the Tahoe host used here)** |
| macOS x86_64 | `macosx_15_0_x86_64` | pip `tensorflow` | ~5 MB; the FIXME disabling `test-command` can be removed |

Free-threaded (`*t-*`) builds stay skipped for the separate reason already documented in
`cibuildwheel-tensorflow.toml`, and cp38 stays skipped.

### Implementation beads a replan should file

1. **Teach the build to find TensorFlow in a pip wheel.** Emit a `tensorflow.pc` (or extend
   the waf check) pointing `includedir` at `site-packages/tensorflow/include` and `libdir`
   at a link directory of `libtensorflow_cc.so`/`libtensorflow_framework.so` symlinks. Add
   `-Wl,-headerpad_max_install_names` plus the two rpaths to `LINKFLAGS`.
2. **Rework `cibuildwheel-tensorflow.toml`.** Replace `brew install tensorflow` with
   `pip install tensorflow==<pin>`; drop `MACOSX_DEPLOYMENT_TARGET=26.2` back to 15.0 for
   both macOS jobs; restore the x86_64 `test-command` and delete the FIXME.
3. **Wire the repair steps.** `repair-wheel-command` with the `--exclude` flags on both
   platforms; keep the existing SDL2 copy step for arm64.
4. **Declare the runtime dependency.** Add `tensorflow` to `dependencies` for the
   `essentia-tensorflow` name only, alongside the existing `sed` package-name patch, and
   document the contract change in the release notes.
5. **Add the linux/aarch64 job.** `ubuntu-24.04-arm` in the workflow matrix, building
   `cp**-manylinux_aarch64` with the `manylinux_2_28_aarch64` image; this is the bead that
   MTG#1486 is really asking for.
6. **Add an installed-wheel smoke test.** Run `TensorflowPredictEffnetDiscogs` against
   `discogs-effnet-bs64-1.pb` in `test-command`, asserting the `(N, 1280)` embedding shape,
   so a broken loader path fails CI instead of shipping.
7. **Fallback bead (only if 4 is rejected):** vendor from the official
   `libtensorflow-cpu-darwin-arm64` 2.17.0 tarball on macOS and the Homebrew `arm64_linux`
   bottle on linux, keeping the self-contained contract and accepting the 2.18.0 ceiling on
   the darwin channel.

## Postscript (2026-09-07): the Homebrew floor is fixed; macOS arm64 moved to the bottle

The measurements above stand as taken, but the premise of Candidate C's NO-GO no longer
holds. The 26.2 floor was a formula bug, reported as
[Homebrew/homebrew-core#302034](https://github.com/Homebrew/homebrew-core/issues/302034)
and fixed by [Homebrew/homebrew-core#302294](https://github.com/Homebrew/homebrew-core/pull/302294)
(merged 2026-09-07): Bazel's Apple toolchain passes an explicit `-target` whose macOS
version defaults to the SDK, so every bottle carried its builder's SDK as `minos`. The
formula now passes `--macos_minimum_os=#{MacOS.version}`, and the rebuilt 2.21.0 bottles
(`rebuild 1`) measure, again from `LC_BUILD_VERSION` after pulling from ghcr.io:

| bottle tag | minos | sdk | notes |
| --- | --- | --- | --- |
| `arm64_sequoia` (`4646a8c4…0690f176`) | **15.0** on all 9 Mach-O files | 26.2 | replaces the Google 2.18.1 tarball on macOS arm64 |
| `arm64_linux` (`b761769c…9293484444`) | max GLIBC 2.27 | — | unchanged floor; pin bumped from the pre-rebuild digest |
| `sonoma` (x86_64) | not published | — | the rebuild dropped the Intel bottle; macOS x86_64 stays on the Google 2.16.2 tarball |

Two things the bottle route needs that the tarball did not, both handled in
`packaging/fetch_libtensorflow.sh`:

- The macOS bottle is `cellar: :any`, so its dylib ids are
  `@@HOMEBREW_PREFIX@@/opt/libtensorflow/lib/<name>`, a placeholder `brew` rewrites at pour
  time. Linked as-is, the extension records the placeholder and delocate cannot resolve it.
  The script rewrites each id to `<prefix>/lib/<name>` with `install_name_tool -id`.
- `install_name_tool` invalidates the ad-hoc signature, and the arm64 kernel kills a process
  that loads an invalidly signed dylib (verified: a smoke program exits 137 before, prints
  `TF_Version: 2.21.0` after `codesign --force --sign -`). `libtensorflow.2.dylib` in this
  build does not load `libtensorflow_framework` at all, so only the ids change.

Fetching by digest rather than `brew install` is what keeps this from regressing: the pin
names one bottle whose floor was measured, and a future rebuild cannot move it. The
`arm64_sonoma` bottle (minos 14.0) was not chosen because the other Homebrew audio bottles
on the `macos-15` runner already set the wheel floor to 15.0, so it would buy nothing.
