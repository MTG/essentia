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

#include "audiocontext.h"
#include <iostream> // for warning cout

using namespace std;
using namespace essentia;

AudioContext::AudioContext()
  : _isOpen(false), _avStream(0), _muxCtx(0), _codecCtx(0),
    _inputBufSize(0), _buffer(0), _convertCtxAv(0) {
  av_log_set_level(AV_LOG_QUIET); // choices: {AV_LOG_VERBOSE, AV_LOG_QUIET}
  
  // Note: av_register_all() was deprecated and removed in FFmpeg 4.0
  // Modern FFmpeg automatically registers formats and codecs

  if (sizeof(float) != av_get_bytes_per_sample(AV_SAMPLE_FMT_FLT)) {
    throw EssentiaException("Unsupported float size");
  }
}

int AudioContext::create(const std::string& filename,
                         const std::string& format,
                         int nChannels, int sampleRate, int bitrate) {
  if (_muxCtx != 0) close();

  _filename = filename;

  const AVOutputFormat* av_output_format = av_guess_format(format.c_str(), 0, 0);
  if (!av_output_format) {
    throw EssentiaException("Could not find a suitable output format for \"", filename, "\"");
  }
  if (format != av_output_format->name) {
    E_WARNING("Essentia is using a different format than the one supplied. Format used is " << av_output_format->name);
  }

  _muxCtx = avformat_alloc_context();
  if (!_muxCtx) throw EssentiaException("Could not allocate the format context");

  _muxCtx->oformat = const_cast<AVOutputFormat*>(av_output_format);

  // Find encoder first
  const AVCodec* audioCodec = avcodec_find_encoder(av_output_format->audio_codec);
  if (!audioCodec) {
    // Try fallback: use codec id reported by format (older FFmpeg may set this)
    audioCodec = avcodec_find_encoder(_muxCtx->oformat->audio_codec);
  }
  if (!audioCodec) {
    throw EssentiaException("Codec for ", format, " files not found or not supported");
  }

  // Create audio stream and pass the codec to help FFmpeg initialize defaults
  _avStream = avformat_new_stream(_muxCtx, audioCodec);
  if (!_avStream) throw EssentiaException("Could not allocate stream");

  // Create codec context
  _codecCtx = avcodec_alloc_context3(audioCodec);
  if (!_codecCtx) throw EssentiaException("Could not allocate codec context");

  // Set codec context fields
  _codecCtx->codec_id = audioCodec->id;
  _codecCtx->codec_type = AVMEDIA_TYPE_AUDIO;
  _codecCtx->bit_rate = bitrate;
  _codecCtx->sample_rate = sampleRate;
  
  // channel layout
  av_channel_layout_default(&_codecCtx->ch_layout, nChannels);
  // set time_base for codec (1/sample_rate)
  _codecCtx->time_base = AVRational{1, sampleRate};

  // Choose a sample format: prefer common defaults but check codec supports it
  enum AVSampleFormat desired_fmt = AV_SAMPLE_FMT_S16;
  if (audioCodec->id == AV_CODEC_ID_VORBIS) desired_fmt = AV_SAMPLE_FMT_FLTP;
  if (audioCodec->id == AV_CODEC_ID_MP3) desired_fmt = AV_SAMPLE_FMT_S16P; // keep MP3 as planar s16 if desired

  // If codec provides supported list, pick one from it (prefer desired_fmt)
  if (audioCodec->sample_fmts) {
    const enum AVSampleFormat* p = audioCodec->sample_fmts;
    bool found = false;
    while (*p != AV_SAMPLE_FMT_NONE) {
      if (*p == desired_fmt) { found = true; break; }
      ++p;
    }
    if (!found) {
      // fallback to first supported format
      desired_fmt = audioCodec->sample_fmts[0];
    }
  }
  _codecCtx->sample_fmt = desired_fmt;

  // Open codec
  int result = avcodec_open2(_codecCtx, audioCodec, NULL);
  if (result < 0) {
    char errstring[1204];
    av_strerror(result, errstring, sizeof(errstring));
    ostringstream msg;
    msg << "AudioWriter: Could not open codec \"" << audioCodec->long_name << "\" for " << format << " files: " << errstring;
    throw EssentiaException(msg);
  }

  // Copy codec parameters to muxer stream (modern API)
  result = avcodec_parameters_from_context(_avStream->codecpar, _codecCtx);
  if (result < 0) {
    char errstring[1204];
    av_strerror(result, errstring, sizeof(errstring));
    ostringstream msg;
    msg << "Failed to copy codec parameters: " << errstring;
    throw EssentiaException(msg);
  }

  // Ensure stream is marked as audio and set a sensible time_base for muxer
  _avStream->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;
  // Set stream time_base to match codec time base (or 1/sample_rate)
  _avStream->time_base = _codecCtx->time_base;

  // Determine frame_size fallback for PCM codecs (some PCM codecs do not set frame_size)
  switch (_codecCtx->codec_id) {
    case AV_CODEC_ID_PCM_S16LE:
    case AV_CODEC_ID_PCM_S16BE:
    case AV_CODEC_ID_PCM_U16LE:
    case AV_CODEC_ID_PCM_U16BE:
      // use a default input frame size in samples
      _codecCtx->frame_size = 4096 / (_codecCtx->ch_layout.nb_channels * av_get_bytes_per_sample(AV_SAMPLE_FMT_S16));
      break;
    default:
      // for encoders that set frame_size, keep it
      if (_codecCtx->frame_size <= 1) {
        // Some codecs (e.g. vorbis) have variable frame sizes — in that case use a safe default
        if (_codecCtx->codec_id == AV_CODEC_ID_VORBIS || _codecCtx->codec_id == AV_CODEC_ID_FLAC) {
          // vorbis and flac can accept arbitrary nb_samples; choose a reasonable default
          _codecCtx->frame_size = 1024;
        } else {
          throw EssentiaException("Do not know how to encode given format: ", format);
        }
      }
  }

  // Allocate input audio FLT buffer sized for codecCtx->frame_size samples
  _inputBufSize = av_samples_get_buffer_size(NULL,
                                             _codecCtx->ch_layout.nb_channels,
                                             _codecCtx->frame_size,
                                             AV_SAMPLE_FMT_FLT, 0);
  _buffer = (float*)av_malloc(_inputBufSize);
  if (!_buffer) {
    throw EssentiaException("Could not allocate input float buffer");
  }
  
  _pts = 0;  // reset PTS counter for new file

  // Configure sample format conversion
  E_DEBUG(EAlgorithm, "AudioContext: using sample format conversion from libswresample");
  _convertCtxAv = swr_alloc();
  if (!_convertCtxAv) {
    throw EssentiaException("Could not allocate SwrContext");
  }

  av_opt_set_chlayout(_convertCtxAv, "in_chlayout", &_codecCtx->ch_layout, 0);
  av_opt_set_chlayout(_convertCtxAv, "out_chlayout", &_codecCtx->ch_layout, 0);
  av_opt_set_int(_convertCtxAv, "in_sample_rate", _codecCtx->sample_rate, 0);
  av_opt_set_int(_convertCtxAv, "out_sample_rate", _codecCtx->sample_rate, 0);
  av_opt_set_int(_convertCtxAv, "in_sample_fmt", AV_SAMPLE_FMT_FLT, 0);
  av_opt_set_int(_convertCtxAv, "out_sample_fmt", _codecCtx->sample_fmt, 0);

  if (swr_init(_convertCtxAv) < 0) {
    throw EssentiaException("AudioLoader: Could not initialize swresample context");
  }

  return _codecCtx->frame_size;
}


void AudioContext::open() {
  if (_isOpen) return;

  if (!_muxCtx) throw EssentiaException("Trying to open an audio file that has not been created yet or has been closed");
  
  // Open the output IO
  int err = avio_open(&_muxCtx->pb, _filename.c_str(), AVIO_FLAG_WRITE);
  if (err < 0) {
    char errstring[1204];
    av_strerror(err, errstring, sizeof(errstring));
    throw EssentiaException("Could not open \"", _filename, "\": ", errstring);
  }

  // Write header
  err = avformat_write_header(_muxCtx, NULL);
  if (err < 0) {
    char errstring[1204];
    av_strerror(err, errstring, sizeof(errstring));
    throw EssentiaException("Could not write header for \"", _filename, "\": ", errstring);
  }
  
  _isOpen = true;
}


void AudioContext::close() {
  if (!_muxCtx) return;

  // Close output file
  if (_isOpen) {
    // Flush encoder via writeEOF()
    writeEOF();

    // Write trailer to the end of the file
    av_write_trailer(_muxCtx);

    // close output IO
    if (_muxCtx->pb) {
      avio_closep(&_muxCtx->pb);  // modern safe API
    }

    _isOpen = false;
  }

  // Use modern API for codec context cleanup
  if (_codecCtx) {
    avcodec_free_context(&_codecCtx);
    _codecCtx = nullptr;
  }

  // free input buffer
  if (_buffer) {
    av_freep(&_buffer);
    _buffer = nullptr;
  }

  // free swresample context
  if (_convertCtxAv) {
    swr_close(_convertCtxAv);
    swr_free(&_convertCtxAv);
    _convertCtxAv = nullptr;
  }

  if (_muxCtx) {
    avformat_free_context(_muxCtx);
    _muxCtx = nullptr;
    _avStream = nullptr;
  }
}


void AudioContext::write(const vector<StereoSample>& stereoData) {
  if (_codecCtx->ch_layout.nb_channels != 2) {
    throw EssentiaException("Trying to write stereo audio data to an audio file with ", _codecCtx->ch_layout.nb_channels, " channels");
  }

  int dsize = (int)stereoData.size();
  
  if (dsize > _codecCtx->frame_size) {
    // AudioWriter sets up correct buffer sizes in accordance to what 
    // AudioContext:create() returns. Nevertherless, double-check here.
    ostringstream msg;
    msg << "Audio frame size " << _codecCtx->frame_size << 
           " is not sufficent to store " << dsize << " samples";
    throw EssentiaException(msg);
  }

  for (int i=0; i<dsize; ++i) {
    _buffer[2*i] = (float) stereoData[i].left();
    _buffer[2*i+1] = (float) stereoData[i].right();
  }

  encodePacket(dsize);
}


void AudioContext::write(const vector<AudioSample>& monoData) {
  if (_codecCtx->ch_layout.nb_channels != 1) {
    throw EssentiaException("Trying to write mono audio data to an audio file with ", _codecCtx->ch_layout.nb_channels, " channels");
  }

  int dsize = (int)monoData.size();
  if (dsize > _codecCtx->frame_size) {
    // The same as for stereoData version of write()
    ostringstream msg;
    msg << "Audio frame size " << _codecCtx->frame_size << 
           " is not sufficent to store " << dsize << " samples";
    throw EssentiaException(msg);
  }

  for (int i=0; i<dsize; ++i) _buffer[i] = (float) monoData[i];

  encodePacket(dsize);
}


void AudioContext::encodePacket(int size) {

  int tmp_fs = _codecCtx->frame_size;
  if (size < _codecCtx->frame_size) {
    _codecCtx->frame_size = size;
  }
  else if (size > _codecCtx->frame_size) {
    throw EssentiaException("AudioLoader: Input audio segment is larger than the codec's frame size");
  }

  // prepare conversion buffers (bufferFmt[ch]) and linesize[ch]
  uint8_t* bufferFmt[AV_NUM_DATA_POINTERS] = { nullptr };
  int linesize[AV_NUM_DATA_POINTERS] = { 0 };
  AVFrame* frame = nullptr;

  try {
    if (av_samples_alloc(bufferFmt,
                         linesize,
                         _codecCtx->ch_layout.nb_channels,
                         size,
                         _codecCtx->sample_fmt,
                         0) < 0) {
      throw EssentiaException("Could not allocate output buffer for sample format conversion");
    }

    // perform sample format conversion
    int written = swr_convert(_convertCtxAv,
                              bufferFmt,
                              size,
                              (const uint8_t**)&_buffer,
                              size);

    if (written < size) {
      ostringstream msg;
      msg << "AudioLoader: Incomplete format conversion (some samples missing)"
          << " from " << av_get_sample_fmt_name(AV_SAMPLE_FMT_FLT)
          << " to "   << av_get_sample_fmt_name(_codecCtx->sample_fmt);
      av_freep(&bufferFmt[0]);
      throw EssentiaException(msg);
    }

    // allocate frame
    frame = av_frame_alloc();
    if (!frame) {
        av_freep(&bufferFmt[0]);
        throw EssentiaException("Error allocating audio frame");
    }

    frame->nb_samples = _codecCtx->frame_size;
    frame->format = _codecCtx->sample_fmt;
    frame->ch_layout = _codecCtx->ch_layout;

    if (av_frame_get_buffer(frame, 0) < 0) {
      av_frame_free(&frame);
      av_freep(&bufferFmt[0]);
      throw EssentiaException("Could not allocate audio frame buffer");
    }

    // Copy converted audio into AVFrame
    int bytesPerSample = av_get_bytes_per_sample(_codecCtx->sample_fmt);
    if (av_sample_fmt_is_planar(_codecCtx->sample_fmt)) {
      for (int ch = 0; ch < _codecCtx->ch_layout.nb_channels; ++ch) {
        memcpy(frame->data[ch], bufferFmt[ch], size * bytesPerSample);
      }
    } else {
      memcpy(frame->data[0], bufferFmt[0], size * _codecCtx->ch_layout.nb_channels * bytesPerSample);
    }

    // send frame to encoder
    int result = avcodec_send_frame(_codecCtx, frame);
    if (result < 0) {
      av_frame_free(&frame);
      av_freep(&bufferFmt[0]);
      char errstring[1024];
      av_strerror(result, errstring, sizeof(errstring));
      ostringstream msg;
      msg << "Error sending frame to encoder: " << errstring;
      throw EssentiaException(msg);
    }

    // receive packets and write them (may be 0..N packets)
    AVPacket packet;
    av_init_packet(&packet);
    packet.data = NULL;
    packet.size = 0;

    while (result >= 0) {
      result = avcodec_receive_packet(_codecCtx, &packet);
      if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) break;
      else if (result < 0) {
        char errstring[1024];
        av_strerror(result, errstring, sizeof(errstring));
        ostringstream msg;
        msg << "Error receiving packet from encoder: " << errstring;
        throw EssentiaException(msg);
      }

      // ensure stream index set
      packet.stream_index = _avStream->index;

      // assign PTS/DTS
      packet.pts = _pts;
      packet.dts = _pts;
      _pts += frame->nb_samples;

      // write packet (interleaved)
      if (av_write_frame(_muxCtx, &packet) != 0) {
        av_packet_unref(&packet);
        av_frame_free(&frame);
        av_freep(&bufferFmt[0]);
        throw EssentiaException("Error while writing audio frame");
      }

      av_packet_unref(&packet);
    }

    // cleanup
    av_frame_free(&frame);
    av_freep(&bufferFmt[0]);    // frees all planes + internal table
    _codecCtx->frame_size = tmp_fs;
  } catch (...) {
    av_frame_free(&frame);
    av_freep(&bufferFmt[0]);
    throw;
  }
}


void AudioContext::writeEOF() {
  if (!_codecCtx) return;
  // Send NULL frame to flush the encoder
  int result = avcodec_send_frame(_codecCtx, NULL);
  if (result < 0) {
    char errstring[1024];
    av_strerror(result, errstring, sizeof(errstring));
    ostringstream msg;
    msg << "Error flushing encoder: " << errstring;
    throw EssentiaException(msg);
  }

  // Receive all remaining packets from encoder
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = NULL;
  packet.size = 0;

  try {
    while (true) {
      result = avcodec_receive_packet(_codecCtx, &packet);
      if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
        break; // No more packets available
      } else if (result < 0) {
        char errstring[1024];
        av_strerror(result, errstring, sizeof(errstring));
        throw EssentiaException("Error receiving packet from encoder during EOF flush: ", errstring);
      }

      // Assign stream index
      packet.stream_index = _avStream->index;

      // Update ptd and dts packet
      packet.pts = _pts;
      packet.dts = _pts;
      _pts += _codecCtx->frame_size;

      // Write with interleaving
      if (av_write_frame(_muxCtx, &packet) < 0) {
        av_packet_unref(&packet);
        throw EssentiaException("Error while writing delayed audio frame");
      }
      av_packet_unref(&packet);
    }
  } catch (...) {
      av_packet_unref(&packet);  // always free packet
      throw;
  }
}
