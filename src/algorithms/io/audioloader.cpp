/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#include "audioloader.h"
#include "algorithmfactory.h"
#include <iomanip>  //  setw()

using namespace std;

namespace essentia {
namespace streaming {

const char* AudioLoader::name = essentia::standard::AudioLoader::name;
const char* AudioLoader::category = essentia::standard::AudioLoader::category;
const char* AudioLoader::description = essentia::standard::AudioLoader::description;


AudioLoader::~AudioLoader() {
    closeAudioFile();

    av_freep(&_buffer);
    av_freep(&_md5Encoded);
    av_freep(&_decodedFrame);
}

void AudioLoader::configure() {
    // set ffmpeg to be silent by default, so we don't have these annoying
    // "invalid new backstep" messages anymore, when everything is actually fine
    av_log_set_level(AV_LOG_QUIET);   // choices: {AV_LOG_VERBOSE, AV_LOG_QUIET}
    _computeMD5 = parameter("computeMD5").toBool();
    _selectedStream = parameter("audioStream").toInt();

    _startTime = parameter("startTime").toReal();
    _endTime   = parameter("endTime").toReal();
    // Same rule Trimmer has always enforced for the composites that expose these parameters
    // (EasyLoader, EqloudLoader): an inverted range is an error, an EMPTY range is not -- it
    // simply yields no audio. Rejecting startTime == endTime here would break callers that
    // work today.
    if (_endTime < _startTime) {
        throw EssentiaException("AudioLoader: 'startTime' cannot be larger than 'endTime'");
    }

    string gapless = parameter("gapless").toLower();
    if      (gapless == "none")    _gapless = GAPLESS_NONE;
    else if (gapless == "decoder") _gapless = GAPLESS_DECODER;
    else                           _gapless = GAPLESS_METADATA;

    reset();
}


// Samples of latency inherent to an MPEG-1/2 Layer III decoder: the synthesis polyphase
// filterbank and the MDCT overlap together delay the output by 528 + 1 samples before the
// first encoded sample appears. It is a property of the decoder, not of the file, so every
// Layer III decoder (LAME's, and libavcodec's) shows it -- which is why an mp3 that DOES carry
// a Xing/LAME header has 529 added to the encoder delay recorded there.
//
// Measured, not assumed: decoding a 2 s file whose LAME header declares a 576-sample encoder
// delay, ffmpeg drops 1105 = 576 + 529 samples and the result is sample-aligned with the
// original wav. Stripping the header from the same file removes the trim and reintroduces
// exactly those 1105 samples of offset, of which 529 disappear again by dropping this constant.
static const int64_t MPEG_LAYER3_DECODER_DELAY = 529;


// How much audio to decode-and-discard BEFORE the requested startTime.
//
// Seeking a compressed stream lands on a packet boundary, never on a sample, and the packet
// it lands on is not necessarily decodable in isolation: MP3 carries a bit reservoir across
// frames and every MDCT codec needs the previous window to reconstruct the current one. So a
// seek that jumps straight to the target packet produces samples that differ from the ones a
// decode-from-zero would have produced at the same position -- not by much, but audibly and
// measurably, and enough to move a BPM or key estimate.
//
// The fix is the standard one: ask for a position SEEK_PREROLL_SEC earlier than we want, then
// decode forward and throw the result away until the exact target sample. By the time the
// target arrives the decoder has been running long enough for its history to be identical to
// the from-zero case, and the output is bit-exact (verified in the test suite).
//
// 0.5 s is ~19 MP3 frames / ~21 AAC frames, far beyond any codec's actual history, and costs
// well under a millisecond of decode. Making it bigger buys nothing; making it much smaller
// starts to matter for codecs with long overlap windows.
static const Real SEEK_PREROLL_SEC = 0.5;


// Position the demuxer at (startTime - preroll) and put the decoder in a clean state.
//
// This is the whole of the actual "seek". It is deliberately conservative: AVSEEK_FLAG_BACKWARD
// guarantees ffmpeg lands at or BEFORE the requested timestamp (never after, which would lose
// audio the caller asked for), and if the seek fails for any reason we simply do not seek --
// _startSample trimming in copyFFmpegOutput() then degrades to the old decode-and-discard
// behaviour, which is slow but still CORRECT. A format with a broken index gets the answer it
// always got, not an error.
void AudioLoader::seekToStartTime() {
    _currentSample   = 0;
    _reachedEndTime  = false;
    _anchorPending   = false;

    int sampleRate = _audioCtx->sample_rate;

    // TRUNCATION, not rounding, and in double: this is exactly Trimmer's seconds -> samples
    // rule, which is what EasyLoader/EqloudLoader have always used to cut their slices. Keeping
    // the same rule is what allows those composites to switch from decode-and-trim to seeking
    // without moving a single sample. (Trimmer does the multiplication in Real, i.e. float32,
    // whose 24-bit mantissa cannot represent every sample index past ~380 s at 44.1 kHz; doing
    // it in double here removes that drift, so a slice taken deep into a long file lands where
    // the parameters say it should.)
    // _decoderDelay shifts the whole requested window: in GAPLESS_DECODER mode we have decided
    // that the stream's true sample 0 lies that many samples into what the decoder hands us, so
    // "second T" is at T*rate + _decoderDelay of decoder output. Both bounds move together, and
    // the trimming itself then falls out of the existing skip logic in copyFFmpegOutput().
    _startSample = (int64_t)((double)_startTime * sampleRate) + _decoderDelay;
    _endSample   = (_endTime >= 1.0e6) ? -1 : (int64_t)((double)_endTime * sampleRate) + _decoderDelay;

    // Seek only when the caller actually asked for a later start. A _decoderDelay of a few
    // hundred samples is not worth a seek -- copyFFmpegOutput() drops it from the first frames.
    if (_startTime <= 0) return;

    // gapless="none" deliberately keeps the encoder delay, so its sample 0 is the encoder delay
    // EARLIER than the timeline the frame timestamps are stated on -- anchoring on a pts after a
    // seek would place the slice by that much. Positions in this mode mean positions in the raw
    // decoder output, so take the decode-and-discard path below instead: slower, but it counts
    // the samples we actually emit and so lands exactly where the parameters say.
    if (_gapless == GAPLESS_NONE) return;

    AVStream* stream = _demuxCtx->streams[_streamIdx];

    Real seekTime = _startTime - SEEK_PREROLL_SEC;
    if (seekTime < 0) seekTime = 0;

    int64_t startOffset = (stream->start_time == AV_NOPTS_VALUE) ? 0 : stream->start_time;
    int64_t target = startOffset + (int64_t)(seekTime / av_q2d(stream->time_base) + 0.5);

    int errnum = av_seek_frame(_demuxCtx, _streamIdx, target, AVSEEK_FLAG_BACKWARD);
    if (errnum < 0) {
        // Not fatal: fall back to decoding from the beginning and discarding. Slow, correct.
        E_WARNING("AudioLoader: seek to " << seekTime << "s failed; "
                  "falling back to decoding from the start of the stream");
        return;
    }

    avcodec_flush_buffers(_audioCtx);
    if (_convertCtxAv) swr_init(_convertCtxAv);   // drop any samples held by the converter

    // We no longer know where we are; the next decoded frame's pts tells us.
    _anchorPending = true;
}


// Adopt the freshly decoded frame's presentation timestamp as our absolute sample position.
//
// Called once after each seek. Converting pts -> sample index is what makes the result
// SAMPLE-ACCURATE rather than packet-accurate: we learn exactly how far before the target the
// seek actually landed, and copyFFmpegOutput() then discards exactly that many samples.
void AudioLoader::anchorPositionFromFrame() {
    AVStream* stream = _demuxCtx->streams[_streamIdx];

    int64_t pts = _decodedFrame->pts;
    if (pts == AV_NOPTS_VALUE) pts = _decodedFrame->best_effort_timestamp;
    if (pts == AV_NOPTS_VALUE) {
        // No timing information at all. We cannot know where we are, so the only safe reading
        // is "we are at the target" -- the caller gets packet-accurate audio rather than a
        // silently shifted stream. Formats reaching this branch would not have been seekable
        // in any useful sense anyway.
        E_WARNING("AudioLoader: no timestamp on the first frame after seeking; "
                  "the slice may not be sample-accurate");
        _currentSample = _startSample;
        _anchorPending = false;
        return;
    }

    int64_t startOffset = (stream->start_time == AV_NOPTS_VALUE) ? 0 : stream->start_time;
    _currentSample = av_rescale_q(pts - startOffset, stream->time_base,
                                  (AVRational){1, _audioCtx->sample_rate});


    if (_currentSample < 0) _currentSample = 0;
    _anchorPending = false;
}


void AudioLoader::openAudioFile(const string& filename) {
    E_DEBUG(EAlgorithm, "AudioLoader: opening file: " << filename);

    // Open file
    int errnum;
    if ((errnum = avformat_open_input(&_demuxCtx, filename.c_str(), NULL, NULL)) != 0) {
        char errorstr[128];
        string error = "Unknown error";
        if (av_strerror(errnum, errorstr, 128) == 0) error = errorstr;
        throw EssentiaException("AudioLoader: Could not open file \"", filename, "\", error = ", error);
    }

    // Retrieve stream information
    if ((errnum = avformat_find_stream_info(_demuxCtx, NULL)) < 0) {
        char errorstr[128];
        string error = "Unknown error";
        if (av_strerror(errnum, errorstr, 128) == 0) error = errorstr;
        avformat_close_input(&_demuxCtx);
        _demuxCtx = 0;
        throw EssentiaException("AudioLoader: Could not find stream information, error = ", error);
    }

    // Check that we have only 1 audio stream in the file
    _streams.clear();
    for (int i=0; i<(int)_demuxCtx->nb_streams; i++) {
        // Use modern API to get codec parameters
        const AVCodecParameters* codecParams = _demuxCtx->streams[i]->codecpar;
        if (codecParams->codec_type == AVMEDIA_TYPE_AUDIO) {
            _streams.push_back(i);
        }
    }
    int nAudioStreams = _streams.size();
    
    if (nAudioStreams == 0) {
        avformat_close_input(&_demuxCtx);
        _demuxCtx = 0;
        throw EssentiaException("AudioLoader ERROR: found 0 streams in the file, expecting one or more audio streams");
    }

    if (_selectedStream >= nAudioStreams) {
        avformat_close_input(&_demuxCtx);
        _demuxCtx = 0;
        throw EssentiaException("AudioLoader ERROR: 'audioStream' parameter set to ", _selectedStream ,". It should be smaller than the audio streams count, ", nAudioStreams);
    }

    _streamIdx = _streams[_selectedStream];

    // Create codec context from stream parameters (modern approach)
    const AVCodecParameters* codecParams = _demuxCtx->streams[_streamIdx]->codecpar;
    _audioCodec = avcodec_find_decoder(codecParams->codec_id);

    if (!_audioCodec) {
        throw EssentiaException("AudioLoader: Unsupported codec!");
    }

    _audioCtx = avcodec_alloc_context3(_audioCodec);
    if (!_audioCtx) {
        throw EssentiaException("AudioLoader: Could not allocate codec context");
    }

    // Copy parameters from stream to codec context
    if (avcodec_parameters_to_context(_audioCtx, codecParams) < 0) {
        avcodec_free_context(&_audioCtx);
        throw EssentiaException("AudioLoader: Could not copy codec parameters");
    }

    // Tell the decoder what time base the packets we feed it are in. libavcodec needs this to
    // report frame timestamps correctly -- in particular it advances frame->pts by however many
    // samples it skipped (codec delay / pre-skip), and that adjustment is expressed in
    // pkt_timebase. Left unset the adjustment is computed against an invalid time base, so
    // frame->pts is wrong for every format that carries an encoder delay.
    _audioCtx->pkt_timebase = _demuxCtx->streams[_streamIdx]->time_base;

    // Gapless trimming is libavcodec's default: the demuxer reads the encoder delay and padding
    // out of the container (the Xing/LAME header for mp3, the edit list for m4a, the pre-skip
    // for opus) and hands them to the decoder as AV_PKT_DATA_SKIP_SAMPLES side data, which the
    // decoder applies before we ever see the frame. AV_CODEC_FLAG2_SKIP_MANUAL turns that off
    // and gives us the raw output instead, which is what gapless="none" asks for.
    if (_gapless == GAPLESS_NONE) {
        _audioCtx->flags2 |= AV_CODEC_FLAG2_SKIP_MANUAL;
    }

    if (avcodec_open2(_audioCtx, _audioCodec, NULL) < 0) {
        avcodec_free_context(&_audioCtx);
        throw EssentiaException("AudioLoader: Unable to instantiate codec...");
    }
  
    // Configure format conversion (no samplerate conversion yet)
    // Use modern channel layout API
    AVChannelLayout layout;
    if (_audioCtx->ch_layout.nb_channels > 0) {
        layout = _audioCtx->ch_layout;
    } else {
        // Fallback for older codecs that might not have channel layout set
        av_channel_layout_default(&layout, _audioCtx->ch_layout.nb_channels);
    }

    E_DEBUG(EAlgorithm, "AudioLoader: using sample format conversion from libswresample");
    _convertCtxAv = swr_alloc();
        
    // Use modern channel layout API for swresample configuration
    av_opt_set_chlayout(_convertCtxAv, "in_chlayout", &layout, 0);
    av_opt_set_chlayout(_convertCtxAv, "out_chlayout", &layout, 0);
    av_opt_set_int(_convertCtxAv, "in_sample_rate", _audioCtx->sample_rate, 0);
    av_opt_set_int(_convertCtxAv, "out_sample_rate", _audioCtx->sample_rate, 0);
    av_opt_set_int(_convertCtxAv, "in_sample_fmt", _audioCtx->sample_fmt, 0);
    av_opt_set_int(_convertCtxAv, "out_sample_fmt", AV_SAMPLE_FMT_FLT, 0);

    if (swr_init(_convertCtxAv) < 0) {
        throw EssentiaException("AudioLoader: Could not initialize swresample context");
    }

    av_init_packet(&_packet);

    _decodedFrame = av_frame_alloc();
    if (!_decodedFrame) {
        throw EssentiaException("AudioLoader: Could not allocate audio frame");
    }

    av_md5_init(_md5Encoded);

    // An mp3 that carries a Xing/LAME header gets its encoder delay reported by the demuxer as a
    // non-zero stream start_time (it is the same number the decoder skips), so a start_time of
    // zero on an mp3 means the file declares nothing and libavcodec has trimmed nothing. That is
    // the only case where gapless="decoder" has anything to add: the encoder delay is
    // unrecoverable without the header, but the decoder's own latency is a constant we know.
    // Restricted to Layer III on purpose -- the constant is Layer III's, not Layer I's or II's.
    _decoderDelay = 0;
    if (_gapless == GAPLESS_DECODER) {
        const AVStream* stream = _demuxCtx->streams[_streamIdx];
        bool delayDeclared = (stream->start_time != AV_NOPTS_VALUE && stream->start_time > 0);
        if (_audioCtx->codec_id == AV_CODEC_ID_MP3 && !delayDeclared) {
            _decoderDelay = MPEG_LAYER3_DECODER_DELAY;
        }
    }

    // Everything above is stream setup; the file is now positioned at the start. If the caller
    // asked for a later startTime, move there NOW rather than making process() read past it.
    seekToStartTime();
}


void AudioLoader::closeAudioFile() {
    if (!_demuxCtx) {
        return;
    }

    if (_convertCtxAv) {
        swr_close(_convertCtxAv);
        swr_free(&_convertCtxAv);
    }

    // Close the codec using modern API
    if (_audioCtx) {
        avcodec_free_context(&_audioCtx);
    }
    // Close the audio file
    if (_demuxCtx) avformat_close_input(&_demuxCtx);

    // free AVPacket using modern API
    // TODO: use a variable for whether _packet is initialized or not
    av_packet_unref(&_packet);
    _demuxCtx = 0;
    _audioCtx = 0;
    _streams.clear();
}


void AudioLoader::pushChannelsSampleRateInfo(int nChannels, Real sampleRate) {
    if (nChannels > 2) {
        throw EssentiaException("AudioLoader: could not load audio. Audio file has more than 2 channels.");
    }
    if (sampleRate <= 0) {
        throw EssentiaException("AudioLoader: could not load audio. Audio sampling rate must be greater than 0.");
    }

    _nChannels = nChannels;

    _channels.push(nChannels);
    _sampleRate.push(sampleRate);
}


void AudioLoader::pushCodecInfo(std::string codec, int bit_rate) {
    _codec.push(codec);
    _bit_rate.push(bit_rate);
}


string uint8_t_to_hex(uint8_t* input, int size) {
    ostringstream result;
    for(int i=0; i<size; ++i) {
        result << setw(2) << setfill('0') << hex << (int) input[i];
    }
    return result.str();
}


AlgorithmStatus AudioLoader::process() {
    if (!parameter("filename").isConfigured()) {
        throw EssentiaException("AudioLoader: Trying to call process() on an AudioLoader algo which hasn't been correctly configured.");
    }

    // read frames until we get a good one
    do {
        int result = av_read_frame(_demuxCtx, &_packet);
        //E_DEBUG(EAlgorithm, "AudioLoader: called av_read_frame(), got result = " << result);
        if (result != 0) {
            // 0 = OK, < 0 = error or EOF
            if (result != AVERROR_EOF) {
                char errstring[1204];
                av_strerror(result, errstring, sizeof(errstring));
                ostringstream msg;
                msg << "AudioLoader: Error reading frame: " << errstring;
                E_WARNING(msg.str());
            }
            shouldStop(true);
            flushPacket();
            closeAudioFile();
            if (_computeMD5) {
                av_md5_final(_md5Encoded, _checksum);
                _md5.push(uint8_t_to_hex(_checksum, 16));
            }
            else {
                string md5 = "";
                _md5.push(md5);
            }
            return FINISHED;
        }
    } while (_packet.stream_index != _streamIdx);

    // compute md5 first
    if (_computeMD5) {
        av_md5_update(_md5Encoded, _packet.data, _packet.size);
    }
    
    // decode ONE frame from this packet (if any). decodePacket() will
    // *not* mutate _packet.data/_packet.size. It will set _dataSize to number of bytes written.
    int consumed = decodePacket();

    // After decodePacket we may have produced audio in _buffer (bytes in _dataSize).
    if (_dataSize > 0) {
        // copyFFmpegOutput will acquire once and release once
        copyFFmpegOutput();
        // reset _dataSize so we don't accidentally reuse it
        _dataSize = 0;
    }
    
    // needs to be freed using modern API !!
    av_packet_unref(&_packet);

    // endTime reached: stop here rather than reading the rest of the file. This is the half of
    // issue #771 that saves work at the TAIL of a slice, exactly as the seek saves it at the head.
    if (_reachedEndTime) {
        shouldStop(true);
        closeAudioFile();
        // An endTime slice never reads the whole payload, so an MD5 over it would not be the
        // file's MD5. Report the empty string rather than a checksum that means nothing.
        _md5.push(string(""));
        return FINISHED;
    }

    return OK;
}


int AudioLoader::decode_audio_frame(AVCodecContext* audioCtx,
                                    float* output,
                                    int* outputSize,
                                    AVPacket* packet) {

    // _dataSize  input = number of bytes available for write in buff
    //           output = number of bytes actually written (actual: FLT data)
    //E_DEBUG(EAlgorithm, "decode_audio_frame, available bytes in buffer = " << _dataSize);
    int gotFrame = 0;
    av_frame_unref(_decodedFrame);

    // Use modern decoding API: send packet to decoder
    int send_result = avcodec_send_packet(audioCtx, packet);
    if (send_result < 0) return send_result; // error handling should be done outside

    // Receive decoded frame from decoder
    int receive_result = avcodec_receive_frame(audioCtx, _decodedFrame);
    if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF) {
        gotFrame = 0;
        // Return the number of bytes that would have been consumed
        // For flush packets (empty packets), return 0
        return (packet->size > 0) ? packet->size : 0;
    } else if (receive_result < 0) {
        return receive_result; // error handling should be done outside
    }
    gotFrame = 1;

    if (gotFrame) {
        int inputSamples = _decodedFrame->nb_samples;
        int inputPlaneSize = av_samples_get_buffer_size(NULL, _nChannels, inputSamples,
                                                        audioCtx->sample_fmt, 1);
        int outputPlaneSize = av_samples_get_buffer_size(NULL, _nChannels, inputSamples,
                                                        AV_SAMPLE_FMT_FLT, 1);
        // the size of the output buffer in samples
        int outputBufferSamples = *outputSize / 
                (av_get_bytes_per_sample(AV_SAMPLE_FMT_FLT) * _nChannels);

        if (outputBufferSamples < inputSamples) { 
            // this should never happen, throw exception here
            throw EssentiaException("AudioLoader: Insufficient buffer size for format conversion");
        }

        if (audioCtx->sample_fmt == AV_SAMPLE_FMT_FLT) {
            // TODO: no need in this check? Not many of common formats support FLT
            // no conversion needed, direct copy from our frame to output buffer
            memcpy(output, _decodedFrame->data[0], inputPlaneSize);
        }
        else {
          int samplesWrittern = swr_convert(_convertCtxAv,
                                          (uint8_t**) &output, 
                                          outputBufferSamples, 
                                          (const uint8_t**)_decodedFrame->data,
                                          inputSamples);

          if (samplesWrittern < inputSamples) {
              // TODO: there may be data remaining in the internal FIFO buffer
              // to get this data: call swr_convert() with NULL input
              // Test if this happens in practice
              ostringstream msg;
              msg << "AudioLoader: Incomplete format conversion (some samples missing)"
                  << " from " << av_get_sample_fmt_name(_audioCtx->sample_fmt)
                  << " to "   << av_get_sample_fmt_name(AV_SAMPLE_FMT_FLT);
              throw EssentiaException(msg);
          }
        }
        *outputSize = outputPlaneSize;
    }
    else {
      E_DEBUG(EAlgorithm, "AudioLoader: tried to decode packet but didn't get any frame...");
      *outputSize = 0;
    }

    // Return the number of bytes consumed from the packet
    // For the modern API, we consume the entire packet when we send it
    return packet->size;
}

void AudioLoader::flushPacket() {
    // Sending a NULL packet tells the decoder to flush internal buffers
    av_packet_unref(&_packet);
    AVPacket empty;
    av_init_packet(&empty);
    empty.data = NULL;
    empty.size = 0;

    // keep draining until decoder stops returning frames
    while (true) {
        _dataSize = 0;
        int send_result = avcodec_send_packet(_audioCtx, &empty);
        if (send_result < 0 && send_result != AVERROR(EAGAIN)) {
            break;
        }
        int receive_result = avcodec_receive_frame(_audioCtx, _decodedFrame);
        if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF) {
            break;
        } else if (receive_result < 0) {
            break;
        }

        // got a frame -> convert to floats as in decodePacket()
        int inputSamples = _decodedFrame->nb_samples;
        int outPlaneSize = av_samples_get_buffer_size(NULL, _nChannels, inputSamples, AV_SAMPLE_FMT_FLT, 1);
        if (outPlaneSize > 0) {
            if (_audioCtx->sample_fmt == AV_SAMPLE_FMT_FLT) {
                memcpy(_buffer, _decodedFrame->data[0], std::min(outPlaneSize, FFMPEG_BUFFER_SIZE));
                _dataSize = std::min(outPlaneSize, FFMPEG_BUFFER_SIZE);
            } else {
                float* outBuff = (float*)_buffer;
                int samplesWritten = swr_convert(_convertCtxAv,
                                                 (uint8_t**)&outBuff,
                                                 inputSamples,
                                                 (const uint8_t**)_decodedFrame->data,
                                                 inputSamples);
                if (samplesWritten > 0) {
                    _dataSize = std::min(samplesWritten * _nChannels * av_get_bytes_per_sample(AV_SAMPLE_FMT_FLT),
                                         FFMPEG_BUFFER_SIZE);
                }
            }
        }

        if (_dataSize > 0) {
            copyFFmpegOutput();
            _dataSize = 0;
        }
    }
}


/**
 * Gets the AVPacket stored in _packet, and decodes all the samples it can from it,
 * putting them in _buffer, the total number of bytes written begin stored in _dataSize.
 */

int AudioLoader::decodePacket() {
    // Prepare float-view of the output buffer
    float* outBuff = (float*)_buffer;
    // Default: no bytes produced yet
    _dataSize = 0;

    // Modern API: send the full packet to the decoder once
    int send_result = avcodec_send_packet(_audioCtx, &_packet);
    if (send_result == AVERROR(EAGAIN)) {
        // decoder not ready to accept packet; try receiving frames first
        // but for streaming we simply try to receive a frame below
    } else if (send_result < 0) {
        // fatal decoding error for this packet
        char errstring[1204];
        av_strerror(send_result, errstring, sizeof(errstring));
        E_WARNING("AudioLoader: avcodec_send_packet error: " << errstring);
        return 0;
    }

    // Try to receive ONE frame
    int receive_result = avcodec_receive_frame(_audioCtx, _decodedFrame);
    if (receive_result == AVERROR(EAGAIN) || receive_result == AVERROR_EOF) {
        // No frame ready from this packet
        return 0;
    } else if (receive_result < 0) {
        char errstring[1204];
        av_strerror(receive_result, errstring, sizeof(errstring));
        E_WARNING("AudioLoader: avcodec_receive_frame error: " << errstring);
        return 0;
    }

    // We got a frame -> convert it to float interleaved
    int inputSamples = _decodedFrame->nb_samples;
    // compute expected number of output bytes for these samples
    int outPlaneSize = av_samples_get_buffer_size(NULL, _nChannels, inputSamples, AV_SAMPLE_FMT_FLT, 1);
    if (outPlaneSize <= 0) {
        E_WARNING("AudioLoader: computed non-positive outPlaneSize");
        return 0;
    }

    // Ensure output buffer is large enough
    if (outPlaneSize > FFMPEG_BUFFER_SIZE) {
        // this shouldn't normally happen; guard and shrink to prevent overflow
        ostringstream msg;
        msg << "AudioLoader: required buffer " << outPlaneSize << " exceeds allocated " << FFMPEG_BUFFER_SIZE;
        E_WARNING(msg.str());
        // clamp to buffer size (will avoid overflow but may drop data)
    }

    // Perform conversion if needed
    if (_audioCtx->sample_fmt == AV_SAMPLE_FMT_FLT) {
        // direct copy - frame data is interleaved in data[0] for packed formats
        memcpy(outBuff, _decodedFrame->data[0], std::min(outPlaneSize, FFMPEG_BUFFER_SIZE));
    } else {
        // Use swr_convert; use pointer to uint8_t* for API compatibility
        int samplesWritten = swr_convert(_convertCtxAv,
                                         (uint8_t**)&outBuff,
                                         inputSamples,
                                         (const uint8_t**)_decodedFrame->data,
                                         inputSamples);
        if (samplesWritten <= 0) {
            E_WARNING("AudioLoader: swr_convert returned no samples");
            return 0;
        }
        // recompute bytes produced
        outPlaneSize = samplesWritten * _nChannels * av_get_bytes_per_sample(AV_SAMPLE_FMT_FLT);
    }

    // commit produced bytes
    _dataSize = std::min(outPlaneSize, FFMPEG_BUFFER_SIZE);

    // Return number of bytes logically consumed from the packet.
    // With modern API we don't need to tell caller how many bytes consumed;
    // we return the original packet size as a hint (caller will unref packet).
    return _packet.size;
}


void AudioLoader::copyFFmpegOutput() {
    int bytesPerSample = av_get_bytes_per_sample(AV_SAMPLE_FMT_FLT);
    int nsamples = _dataSize / (bytesPerSample * _nChannels);
    if (nsamples == 0) return;

    if (_reachedEndTime) return;

    // This block is where startTime/endTime become sample-accurate. The decoder hands us whole
    // frames; the slice the caller asked for starts and ends inside one. `skip` is how many
    // samples of THIS frame fall before startTime -- after a seek that is however far the seek
    // undershot plus the preroll, and with no seek at all it is the whole decode-and-discard
    // prefix.
    int skip = 0;

    if (_anchorPending) anchorPositionFromFrame();

    if (_currentSample < _startSample) {
        int64_t toSkip = _startSample - _currentSample;
        if (toSkip >= nsamples) {
            // Entirely before the slice: drop the frame and do not touch the output at all.
            _currentSample += nsamples;
            return;
        }
        skip = (int)toSkip;
        _currentSample += toSkip;
        nsamples -= skip;
    }

    if (_endSample >= 0 && _currentSample + nsamples >= _endSample) {
        nsamples = (int)(_endSample - _currentSample);
        _reachedEndTime = true;      // process() turns this into FINISHED
        if (nsamples <= 0) return;
    }

    // acquire necessary data
    bool ok = _audio.acquire(nsamples);
    if (!ok) {
        throw EssentiaException("AudioLoader: could not acquire output for audio");
    }

    vector<StereoSample>& audio = *((vector<StereoSample>*)_audio.getTokens());

    float* fbuf = (float*)_buffer; // interpret buffer as floats for copying

    if (_nChannels == 1) {
        for (int i=0; i<nsamples; i++) {
          audio[i].left() = fbuf[skip + i];
        }
    }
    else { // _nChannels == 2
      for (int i=0; i<nsamples; i++) {
        audio[i].left()  = fbuf[2*(skip + i)];
        audio[i].right() = fbuf[2*(skip + i)+1];
      }
    }

    _currentSample += nsamples;
    _audio.release(nsamples);
}


void AudioLoader::reset() {
    Algorithm::reset();

    if (!parameter("filename").isConfigured()) return;

    string filename = parameter("filename").toString();

    closeAudioFile();
    openAudioFile(filename);

    pushChannelsSampleRateInfo(_audioCtx->ch_layout.nb_channels, _audioCtx->sample_rate);
    pushCodecInfo(_audioCodec->name, _audioCtx->bit_rate);
}

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

const char* AudioLoader::name = "AudioLoader";
const char* AudioLoader::category = "Input/output";
const char* AudioLoader::description = DOC("This algorithm loads the single audio stream contained in a given audio or video file. Supported formats are all those supported by the FFmpeg library including wav, aiff, flac, ogg and mp3.\n"
"\n"
"This algorithm will throw an exception if it was not properly configured which is normally due to not specifying a valid filename. Invalid names comprise those with extensions different than the supported  formats and non existent files. If using this algorithm on Windows, you must ensure that the filename is encoded as UTF-8\n\n"
"Note: ogg files are decoded in reverse phase, due to be using ffmpeg library.\n"
"\n"
"Lossy codecs do not encode an exact number of samples: the encoder prepends a delay and appends "
"padding so that the signal fills a whole number of coded frames, and a decoder that ignores them "
"returns audio that is time-shifted and zero-padded with respect to what was encoded. By default "
"('gapless' = \"metadata\") this algorithm trims exactly the delay and padding the file declares "
"-- the Xing/LAME header of an mp3, the edit list of an m4a, the pre-skip of an opus stream -- so "
"the first output sample is the first sample that was encoded and the output length matches the "
"original. An mp3 written without a Xing/LAME header declares nothing and cannot be fully "
"corrected; 'gapless' = \"decoder\" removes the decoder's own 529-sample latency from those files, "
"leaving only the encoder's share of the shift. 'gapless' = \"none\" reproduces the untrimmed "
"output of a decoder that ignores this metadata altogether.\n"
"\n"
"References:\n"
"  [1] WAV - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Wav\n"
"  [2] Audio Interchange File Format - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Aiff\n"
"  [3] Free Lossless Audio Codec - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Flac\n"
"  [4] Vorbis - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Vorbis\n"
"  [5] MP3 - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Mp3");


void AudioLoader::createInnerNetwork() {
    _loader = streaming::AlgorithmFactory::create("AudioLoader");
    _audioStorage = new streaming::VectorOutput<StereoSample>();

    _loader->output("audio")           >>  _audioStorage->input("data");
    _loader->output("sampleRate")      >>  PC(_pool, "internal.sampleRate");
    _loader->output("numberChannels")  >>  PC(_pool, "internal.numberChannels");
    _loader->output("md5")             >>  PC(_pool, "internal.md5");
    _loader->output("codec")           >>  PC(_pool, "internal.codec");
    _loader->output("bit_rate")        >>  PC(_pool, "internal.bit_rate");
    _network = new scheduler::Network(_loader);
}

void AudioLoader::configure() {
    _loader->configure(INHERIT("filename"),
                       INHERIT("computeMD5"),
                       INHERIT("audioStream"),
                       INHERIT("startTime"),
                       INHERIT("endTime"),
                       INHERIT("gapless"));
}

void AudioLoader::compute() {
    if (!parameter("filename").isConfigured()) {
        throw EssentiaException("AudioLoader: Trying to call compute() on an "
                                "AudioLoader algo which hasn't been correctly configured.");
    }

    Real& sampleRate = _sampleRate.get();
    int& numberChannels = _channels.get();
    string& md5 = _md5.get();
    int& bit_rate = _bit_rate.get();
    string& codec = _codec.get();
    vector<StereoSample>& audio = _audio.get();

    _audioStorage->setVector(&audio);
    // TODO: is using VectorInput indeed faster than using Pool?

    // FIXME:
    // _audio.reserve(sth_meaningful);

    _network->run();

    sampleRate = _pool.value<Real>("internal.sampleRate");
    numberChannels = (int) _pool.value<Real>("internal.numberChannels");
    md5 = _pool.value<std::string>("internal.md5");
    bit_rate = (int) _pool.value<Real>("internal.bit_rate");
    codec = _pool.value<std::string>("internal.codec");

    // reset, so it is ready to load audio again
    reset();
}

void AudioLoader::reset() {
    _network->reset();
    _pool.remove("internal.md5");
    _pool.remove("internal.sampleRate");
    _pool.remove("internal.numberChannels");
    _pool.remove("internal.codec");
    _pool.remove("internal.bit_rate");
}

} // namespace standard
} // namespace essentia
