# Phase 3 — Implementation Spec (Batch C): AudioLoader decode path

**Objective:** eliminate the heap-overflow in the sample-format conversion path, recover the samples silently dropped
for multi-frame lossy formats, fix the channel-layout handling, and collapse three diverging copies of the
decode/convert logic into one. This is the highest-risk, security-critical phase — treat CP3.1 as a shippable hotfix.

**Phase models:** Code **Opus 4.8** / Review **Fable 5 (high)** — heap-overflow + FFmpeg send/receive state-machine
semantics are exactly where a plausible-but-wrong fix hides; the reviewer must independently re-derive the buffer math
and the drain loop against the FFmpeg API contract.

**Branch:** single feature branch `fix/audioloader-decode-path`; checkpoints land **in order** on it (CP3.2 depends on
CP3.1's clamp helper; CP3.4 refactors what CP3.1–CP3.3 leave). CP3.1 may additionally be cherry-picked to a hotfix
branch off `master` and released ahead of the rest.

**Prerequisite:** Phase 0 oracles (CP0.2) — the `ffmpeg -i f -f f32le -` full-decode comparator and a crafted-file
generator — must exist before CP3.1/CP3.2 validation.

**Key facts verified against the working tree (re-confirm before editing — lines drift):**
- `_buffer = av_malloc(FFMPEG_BUFFER_SIZE)`, `FFMPEG_BUFFER_SIZE = MAX_AUDIO_FRAME_SIZE * 2` (`audioloader.h:49,101`).
- The FLT **memcpy** path is already clamped (`std::min(outPlaneSize, FFMPEG_BUFFER_SIZE)`), lines 457 / 376.
- The **swr_convert** path is **not** clamped: `decodePacket` line 460-464 and `flushPacket` line 380-384 pass
  `inputSamples` as the output-sample count — libswresample writes `inputSamples * _nChannels * 4` bytes into the
  fixed buffer. This is the overflow.
- `decode_audio_frame` (`cpp:272-347`, decl `h:76`) is **dead** — grep shows only a comment reference; it is the only
  copy carrying the pre-conversion buffer-size guard (lines 309-312).
- `decodePacket` receives exactly **one** frame per packet (single `avcodec_receive_frame` at line 425); `process()`
  then unrefs the packet (line 266). Its `EAGAIN`-on-send branch (413-415) is a comment-only no-op.
- Channel-layout fallback (`cpp:126-131`) passes `nb_channels` (which is `<= 0` in that branch) to
  `av_channel_layout_default`, and the normal branch does a struct-copy `layout = _audioCtx->ch_layout`.

Global acceptance gate for the phase:
```
python test/src/unittests/all_tests.py io base standard rhythm highlevel machinelearning audioproblems
```
green, **plus** each CP's reference comparison below.

---

## CP3.1 — SEC-01: clamp / drain the swr_convert output (heap overflow)
**Findings:** SEC-01  ·  **Severity:** High (exploitable heap write on a crafted file)

### The defect
When a decoded frame's `inputSamples` is large enough that `outPlaneSize > FFMPEG_BUFFER_SIZE`, `decodePacket` only
emits an `E_WARNING` (line 446-452) and then calls `swr_convert(..., outBuff, inputSamples, ...)` — writing past the
end of `_buffer`. A crafted FLAC/format whose decoder emits a huge-`nb_samples` frame overflows the heap. `flushPacket`
has the identical unclamped swr call.

### Fix
Introduce a single helper that both call sites (and CP3.4's unified `convertFrameToBuffer`) use, converting **at most**
what the buffer holds and draining swr's internal FIFO in a loop so valid oversized frames still load fully rather than
being truncated:

```cpp
// maximum interleaved sample-frames that fit in _buffer
const int maxOutSamples = FFMPEG_BUFFER_SIZE /
                          (av_get_bytes_per_sample(AV_SAMPLE_FMT_FLT) * _nChannels);

// First conversion pass: feed the frame, request no more than the buffer holds.
int produced = swr_convert(_convertCtxAv,
                           (uint8_t**)&outBuff, maxOutSamples,
                           (const uint8_t**)_decodedFrame->data, inputSamples);
if (produced < 0) { E_WARNING("AudioLoader: swr_convert failed"); return 0; }
while (produced > 0) {
    _dataSize = produced * _nChannels * av_get_bytes_per_sample(AV_SAMPLE_FMT_FLT);
    // _dataSize <= FFMPEG_BUFFER_SIZE by construction of maxOutSamples
    copyFFmpegOutput();
    _dataSize = 0;
    // Drain any samples swr buffered internally (NULL input), still clamped.
    produced = swr_convert(_convertCtxAv,
                           (uint8_t**)&outBuff, maxOutSamples,
                           (const uint8_t**)NULL, 0);
}
```

Notes for the implementer:
- The FLT **memcpy** path is already safe but should also chunk if `outPlaneSize > FFMPEG_BUFFER_SIZE` (a packed FLT
  frame larger than the buffer): loop `memcpy`/`copyFFmpegOutput` over `min(remaining, FFMPEG_BUFFER_SIZE)` slices.
  In practice FLT-native frames are small, but the guard belongs there for symmetry — do it in the shared helper.
- Because `copyFFmpegOutput` acquires/releases the stream output per chunk, draining in a loop is safe with the
  streaming buffer.
- Remove the misleading "clamp ... may drop data" warning comment once the drain loop makes truncation impossible.
- `maxOutSamples` is constant per configuration — compute it once (member set in `openAudioFile`) rather than per
  frame if the reviewer prefers; either is fine.

Apply the same helper to `flushPacket`.

### Validation
- New crafted-file test in `test/src/unittests/io/test_audioloader.py`: synthesize (ffmpeg CLI) a FLAC with maximum
  block size and high channel count so a single frame exceeds `FFMPEG_BUFFER_SIZE`; assert **either** a full correct
  decode **or** a clean `RuntimeError` — never a crash / ASan report.
- Build with AddressSanitizer for this test if the lane supports it.
- All existing fixtures decode **byte-identical** (this CP must not change valid small-frame output).

**PR gate:** `io` suite green; crafted-file test passes; ASan clean; existing decodes byte-identical.

---

## CP3.2 — BUG-02: receive all frames per packet (dropped-sample recovery)
**Findings:** BUG-02  ·  Depends on CP3.1's clamped helper.

### The defect
mp3/aac/ogg decoders emit multiple frames per packet, and `avcodec_send_packet` can return `EAGAIN` (decoder must be
drained before it accepts more). Today `decodePacket` sends once, receives **one** frame, and `process()` unrefs the
packet — every additional frame is lost, and an `EAGAIN` on send silently drops the whole packet. Result: truncated /
sample-count-short decodes for lossy formats.

### Fix
Restructure `decodePacket` into the canonical FFmpeg send/receive loop:

```cpp
int ret = avcodec_send_packet(_audioCtx, &_packet);
if (ret == AVERROR(EAGAIN)) {
    // decoder full: drain first, then this packet must be re-sent by caller logic.
    // Simplest correct form: drain all pending frames, then resend the packet.
    drainFrames();                       // receive loop below, factored out
    ret = avcodec_send_packet(_audioCtx, &_packet);
}
if (ret < 0 && ret != AVERROR_EOF) {
    char e[256]; av_strerror(ret, e, sizeof(e));
    E_WARNING("AudioLoader: avcodec_send_packet: " << e);
    return 0;
}
// Receive every frame this packet produced.
for (;;) {
    ret = avcodec_receive_frame(_audioCtx, _decodedFrame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
    if (ret < 0) { /* warn, break */ }
    convertFrameToBuffer(_decodedFrame);   // shared helper (CP3.1 body + copyFFmpegOutput per chunk)
}
```

Implementation notes:
- `process()` currently checks `_dataSize > 0` and calls `copyFFmpegOutput()` once (lines 257-263). Once the receive
  loop converts+copies each frame internally via the shared helper, remove that single post-call copy from `process()`
  to avoid double-emitting the last frame — verify carefully; this is the most error-prone edit in the phase.
- Keep MD5 update (line 249-251) where it is (per-packet, on raw data — unaffected).
- Ensure `flushPacket` (already a drain loop) and the new in-`decodePacket` drain don't double-drain at EOF; the EOF
  transition is driven by `process()`'s FINISHED path (line 234 calls `flushPacket`).

### Validation
- `test_audioloader.py`: full-decode compare against `ffmpeg -i f -f f32le -` for representative mp3/aac/ogg + a
  lossless wav/flac. **Sample counts must match the reference exactly.** Tolerance: 0 for lossless, 1e-6 for lossy.
- ⚠ This CP **intentionally changes** lossy-format output (recovers previously-dropped samples) — the reference is
  ffmpeg, not the old essentia baseline. Regenerate/annotate any stored lossy baselines with the ffmpeg-diff evidence
  in the PR body. Lossless formats must stay byte-identical to both ffmpeg and the old baseline.

**PR gate:** ffmpeg-reference comparison passes for all format families with exact sample counts; lossless
byte-identical; lossy baselines regenerated with justification.

---

## CP3.3 — BUG-07: channel-layout copy + zero-channel guard
**Findings:** BUG-07  ·  `src/algorithms/io/audioloader.cpp:126-131` (+ swr setup 137-146)

### The defect
- The fallback branch runs only when `ch_layout.nb_channels <= 0`, yet passes that same value to
  `av_channel_layout_default(&layout, 0)` — a request for a 0-channel default.
- The normal branch does `layout = _audioCtx->ch_layout;` — a shallow struct copy. For custom layouts that own
  `u.map` heap memory, FFmpeg requires `av_channel_layout_copy`; a struct copy is UB (and can double-free / dangle).

### Fix
```cpp
AVChannelLayout layout;
av_channel_layout_uninit(&layout);   // safe-init
if (_audioCtx->ch_layout.nb_channels > 0) {
    if (av_channel_layout_copy(&layout, &_audioCtx->ch_layout) < 0)
        throw EssentiaException("AudioLoader: could not copy channel layout");
} else {
    throw EssentiaException("AudioLoader: decoder reported no audio channels");
}
// ... av_opt_set_chlayout(...) for in/out using &layout ...
// after swr_init succeeds (or on the throw paths):
av_channel_layout_uninit(&layout);
```
Notes:
- Guarding the swr setup throws so `layout` is always `uninit`-ed (add to the error paths or use a small RAII wrapper).
- Removing the bogus `av_channel_layout_default(&layout, 0)` fallback turns a silent-wrong/UB path into a clean error.

### Validation
- Existing `test_audioloader.py` **byte-identical** for all current fixtures (they all report `nb_channels > 0`).
- Add a negative test only if a zero-channel fixture is obtainable; otherwise the byte-identical suite + review covers
  it (the UB path is unreachable by valid inputs).

**PR gate:** `io` suite byte-identical; no new leaks (psutil harness stable across repeated open/close).

---

## CP3.4 — DESIGN-01: delete dead path, extract `convertFrameToBuffer`
**Findings:** DESIGN-01  ·  Behavior-preserving refactor; land last.

### The defect
Three near-copies of the decode/convert logic exist: `decode_audio_frame` (dead), `decodePacket`, `flushPacket`. The
divergence is exactly what let SEC-01 exist in only one copy. `decode_audio_frame` is uncalled (grep-verified — only a
comment mentions it).

### Fix
1. Delete `decode_audio_frame` (definition `cpp:272-347` and declaration `audioloader.h:76-77`).
2. Extract the shared, clamped, FIFO-draining conversion (from CP3.1/CP3.2) into one private method:
   ```cpp
   void AudioLoader::convertFrameToBuffer(const AVFrame* frame);  // converts + copyFFmpegOutput per chunk
   ```
   Have both `decodePacket`'s receive loop and `flushPacket`'s drain loop call it. After this, the swr clamp/drain and
   the FLT chunk-copy exist in **exactly one** place.
3. Confirm no other translation unit references the removed symbol (grep `decode_audio_frame` → zero hits post-edit).

### Validation
- Full `test_audioloader.py` unchanged (behavior-preserving); clean build (no unused-function / missing-symbol
  warnings).
- Re-run the CP3.1 crafted-file test and CP3.2 ffmpeg comparison through the unified helper to prove no regression from
  the extraction.

**PR gate:** entire phase's tests green through the single code path; build clean; dead symbol gone.

---

## Phase 3 exit criteria
- CP3.1 merged (and optionally hotfix-released) with a passing crafted-file overflow test under ASan.
- CP3.2's decodes match the ffmpeg reference with exact sample counts across wav/flac/mp3/aac/ogg; lossy baselines
  regenerated with evidence, lossless byte-identical.
- CP3.3 leaves all current fixtures byte-identical; zero-channel/custom-layout UB paths now raise cleanly.
- CP3.4 leaves one decode/convert code path; `decode_audio_frame` removed.
- Full gate suite green on the final merge commit.

## Handoff notes to the reviewer (Fable 5, high)
1. Re-derive `maxOutSamples` and confirm `_dataSize` can never exceed `FFMPEG_BUFFER_SIZE` on **any** path, including
   the FLT memcpy and the FIFO-drain iterations.
2. Verify the CP3.2 change removes the double-copy in `process()` (the old single post-`decodePacket` copy) — this is
   the likeliest correctness slip; check that no frame is emitted twice and none is dropped at packet boundaries.
3. Confirm the EOF/flush interaction: `process()` FINISHED → `flushPacket`; ensure the in-`decodePacket` drain and the
   flush drain cannot both run on the same buffered samples.
4. Confirm every `AVChannelLayout` created is `av_channel_layout_uninit`-ed on all exit paths (CP3.3).
