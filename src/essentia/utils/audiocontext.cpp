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
  av_log_set_level(AV_LOG_VERBOSE);
  //av_log_set_level(AV_LOG_QUIET);
  
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

  _muxCtx->oformat = av_output_format;

  // Create audio stream
  _avStream = avformat_new_stream(_muxCtx, NULL);
  if (!_avStream) throw EssentiaException("Could not allocate stream");

  // Create codec context separately (modern approach)
  _codecCtx = avcodec_alloc_context3(NULL);
  if (!_codecCtx) throw EssentiaException("Could not allocate codec context");

  // Set codec parameters
  _codecCtx->codec_id = _muxCtx->oformat->audio_codec;
  _codecCtx->codec_type = AVMEDIA_TYPE_AUDIO;
  _codecCtx->bit_rate = bitrate;
  _codecCtx->sample_rate = sampleRate;
  
  // Use modern channel layout API
  av_channel_layout_default(&_codecCtx->ch_layout, nChannels);

  // Find encoder
  av_log_set_level(AV_LOG_VERBOSE);
  const AVCodec* audioCodec = avcodec_find_encoder(_codecCtx->codec_id);
  if (!audioCodec) throw EssentiaException("Codec for ", format, " files not found or not supported");

  switch (_codecCtx->codec_id) {
    case AV_CODEC_ID_VORBIS:
      _codecCtx->sample_fmt = AV_SAMPLE_FMT_FLTP;
      break;
    case AV_CODEC_ID_MP3:
      _codecCtx->sample_fmt = AV_SAMPLE_FMT_S16P;
      break;
    default:
      _codecCtx->sample_fmt = AV_SAMPLE_FMT_S16;
  }

  // Check if the hardcoded sample format is supported by the codec
  if (audioCodec->sample_fmts) {
    const enum AVSampleFormat* p = audioCodec->sample_fmts;
    while (*p != AV_SAMPLE_FMT_NONE) {
      if (*p == _codecCtx->sample_fmt) break;
      p++;
    }
    if (*p == AV_SAMPLE_FMT_NONE) {
      // Not supported --> use the first one in the list as default?
      // _codecCtx->sample_fmt = audioCodec->sample_fmts[0];
      ostringstream msg;  
      msg << "AudioWriter: Could not open codec \"" << audioCodec->long_name << "\" for " 
          << format << " files: sample format " << av_get_sample_fmt_name(_codecCtx->sample_fmt) << " is not supported";
      throw EssentiaException(msg);
    }
  }

  // Open codec and store it in _codecCtx. 
  int result = avcodec_open2(_codecCtx, audioCodec, NULL);
  if (result < 0) {
    char errstring[1204];
    av_strerror(result, errstring, sizeof(errstring));

    ostringstream msg;  
    msg << "AudioWriter: Could not open codec \"" << audioCodec->long_name << "\" for " << format << " files: " << errstring;
    throw EssentiaException(msg);
  }

  switch (_codecCtx->codec_id) {
    case AV_CODEC_ID_PCM_S16LE:
    case AV_CODEC_ID_PCM_S16BE:
    case AV_CODEC_ID_PCM_U16LE:
    case AV_CODEC_ID_PCM_U16BE:
      // PCM codecs do not provide frame size in samples, use 4096 bytes on input
      _codecCtx->frame_size = 4096 / _codecCtx->ch_layout.nb_channels / av_get_bytes_per_sample(AV_SAMPLE_FMT_S16);
      break;

    //case AV_CODEC_ID_FLAC:
    //case AV_CODEC_ID_VORBIS:
    //  break;

    default:
      if (_codecCtx->frame_size <= 1) {
        throw EssentiaException("Do not know how to encode given format: ", format);
      }
  }

  // Allocate input audio FLT buffer
  _inputBufSize = av_samples_get_buffer_size(NULL, 
                                             _codecCtx->ch_layout.nb_channels, 
                                             _codecCtx->frame_size, 
                                             AV_SAMPLE_FMT_FLT, 0);
  _buffer = (float*)av_malloc(_inputBufSize);

  // Note: _muxCtx->filename is deprecated, but we'll keep it for now
  // as it's used in the open() method. Modern approach would use avio_open directly.

  // Configure sample format conversion
  E_DEBUG(EAlgorithm, "AudioContext: using sample format conversion from libswresample");
  _convertCtxAv = swr_alloc();
        
  // Use modern channel layout API for swresample configuration
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

  // Open output file
  if (avio_open(&_muxCtx->pb, _filename.c_str(), AVIO_FLAG_WRITE) < 0) {
    throw EssentiaException("Could not open \"", _filename, "\"");
  }

  avformat_write_header(_muxCtx, /* AVDictionary **options */ NULL);
  _isOpen = true;
}


void AudioContext::close() {
  if (!_muxCtx) return;

  // Close output file
  if (_isOpen) {
    writeEOF();

    // Write trailer to the end of the file
    av_write_trailer(_muxCtx);

    avio_close(_muxCtx->pb);
  }

  // Use modern API for codec context cleanup
  if (_codecCtx) {
    avcodec_free_context(&_codecCtx);
  }

  av_freep(&_buffer);

  av_freep(&_avStream);
  av_freep(&_muxCtx); // TODO also must be av_free, not av_freep

  // TODO: need those assignments?
  _muxCtx = 0;
  _avStream = 0;
  _codecCtx = 0;
  _buffer = 0;

  if (_convertCtxAv) {
    swr_close(_convertCtxAv);
    swr_free(&_convertCtxAv);
  }

  _isOpen = false;
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
    // input audio vector does not fit into the codec's buffer
    throw EssentiaException("AudioLoader: Input audio segment is larger than the codec's frame size");
  }

  // convert sample format to the one required by codec
  int inputPlaneSize = av_samples_get_buffer_size(NULL, 
                                                  _codecCtx->ch_layout.nb_channels, 
                                                  size, 
                                                  AV_SAMPLE_FMT_FLT, 0);
  int outputPlaneSize;  
  uint8_t* bufferFmt;

  if (av_samples_alloc(&bufferFmt, &outputPlaneSize, 
                               _codecCtx->ch_layout.nb_channels, size,
                               _codecCtx->sample_fmt, 0) < 0) {
    throw EssentiaException("Could not allocate output buffer for sample format conversion");
  }
 
  int written = swr_convert(_convertCtxAv,
                                   &bufferFmt, 
                                   size, 
                                   (const uint8_t**) &_buffer,
                                   size);

  if (written < size) {
    // The same as in AudioLoader. There may be data remaining in the internal 
    // FIFO buffer to get this data: call swr_convert() with NULL input
    // But we just throw exception instead.
    ostringstream msg;
    msg << "AudioLoader: Incomplete format conversion (some samples missing)"
        << " from " << av_get_sample_fmt_name(AV_SAMPLE_FMT_FLT)
        << " to "   << av_get_sample_fmt_name(_codecCtx->sample_fmt);
    throw EssentiaException(msg);
  }

  AVFrame *frame;
  frame = av_frame_alloc();  
  if (!frame) {
    throw EssentiaException("Error allocating audio frame");
  }

  frame->nb_samples = _codecCtx->frame_size;
  frame->format = _codecCtx->sample_fmt;
  // Use modern channel layout API for AVFrame
  frame->ch_layout = _codecCtx->ch_layout;

  // Use modern API to fill audio frame
  int result = av_frame_get_buffer(frame, 0);
  if (result < 0) {
    char errstring[1204];
    av_strerror(result, errstring, sizeof(errstring));
    ostringstream msg;
    msg << "Could not allocate audio frame buffer: " << errstring;
    throw EssentiaException(msg);
  }

  // Copy the converted audio data to the frame
  if (av_sample_fmt_is_planar(_codecCtx->sample_fmt)) {
    // Planar format
    for (int ch = 0; ch < _codecCtx->ch_layout.nb_channels; ch++) {
      memcpy(frame->data[ch], bufferFmt + ch * outputPlaneSize, outputPlaneSize);
    }
  } else {
    // Interleaved format
    memcpy(frame->data[0], bufferFmt, outputPlaneSize * _codecCtx->ch_layout.nb_channels);
  }

  // Use modern encoding API: send frame to encoder
  result = avcodec_send_frame(_codecCtx, frame);
  if (result < 0) {
    char errstring[1204];
    av_strerror(result, errstring, sizeof(errstring));
    ostringstream msg;
    msg << "Error sending frame to encoder: " << errstring;
    throw EssentiaException(msg);
  }

  // Receive encoded packets from encoder
  AVPacket packet;
  av_init_packet(&packet);
  packet.data = NULL;
  packet.size = 0;

  while (result >= 0) {
    result = avcodec_receive_packet(_codecCtx, &packet);
    if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
      break; // No more packets available
    } else if (result < 0) {
      char errstring[1204];
      av_strerror(result, errstring, sizeof(errstring));
      ostringstream msg;
      msg << "Error receiving packet from encoder: " << errstring;
      throw EssentiaException(msg);
    }

    // Write the packet to the output file
    if (av_write_frame(_muxCtx, &packet) != 0) {
      throw EssentiaException("Error while writing audio frame");
    }
    av_packet_unref(&packet);
  }

  av_frame_free(&frame);
  av_freep(&bufferFmt);
  _codecCtx->frame_size = tmp_fs;
}

void AudioContext::writeEOF() { 
  // Send NULL frame to flush the encoder
  int result = avcodec_send_frame(_codecCtx, NULL);
  if (result < 0) {
    char errstring[1204];
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

  while (result >= 0) {
    result = avcodec_receive_packet(_codecCtx, &packet);
    if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) {
      break; // No more packets available
    } else if (result < 0) {
      char errstring[1204];
      av_strerror(result, errstring, sizeof(errstring));
      ostringstream msg;
      msg << "Error receiving packet from encoder: " << errstring;
      throw EssentiaException(msg);
    }

    // Write the packet to the output file
    if (av_write_frame(_muxCtx, &packet) != 0) {
      throw EssentiaException("Error while writing delayed audio frame");
    }
    av_packet_unref(&packet);
  }
}
