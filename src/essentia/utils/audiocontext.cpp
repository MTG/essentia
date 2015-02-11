/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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


extern "C" {
#include <libavutil/mathematics.h>
}


using namespace std;
using namespace essentia;

const int AudioContext::SAMPLE_SIZE_RATIO = sizeof(int16_t)/sizeof(int8_t);

AudioContext::AudioContext()
  : _isOpen(false), _avStream(0), _muxCtx(0), _codecCtx(0),
    _outputBufSize(0), _inputBufSize(0), _inputBuffer(0), _outputBuffer(0) {
  //av_log_set_level(AV_LOG_VERBOSE);
  av_log_set_level(AV_LOG_QUIET);
  // Register all formats and codecs
  av_register_all(); // this should be done once only..
}


int AudioContext::create(const std::string& filename,
                         const std::string& format,
                         int nChannels, int sampleRate, int bitrate) {

  if (_muxCtx != 0) close();

  _filename = filename;

  AVOutputFormat* av_output_format = av_guess_format(format.c_str(), 0, 0);
  if (!av_output_format) {
    throw EssentiaException("Could not find a suitable output format for \"", filename, "\"");
  }
  if (format != av_output_format->name) {
    cerr << "WARNING: Essentia is using a different format than the one supplied. Format used is " << av_output_format->name << endl;
  }

  _muxCtx = avformat_alloc_context();
  if (!_muxCtx) throw EssentiaException("Could not allocate the format context");

  _muxCtx->oformat = av_output_format;

  // Create audio stream
  _avStream = avformat_new_stream(_muxCtx, NULL);
  if (!_avStream) throw EssentiaException("Could not allocate stream");
  _avStream->id = 1; // necessary? found here: http://sgros.blogspot.com.es/2013/01/deprecated-functions-in-ffmpeg-library.html

  // Load corresponding codec and set it up:
  _codecCtx                 = _avStream->codec;
  _codecCtx->codec_id       = _muxCtx->oformat->audio_codec;
  _codecCtx->codec_type     = AVMEDIA_TYPE_AUDIO;
  _codecCtx->sample_fmt     = AV_SAMPLE_FMT_S16;
  _codecCtx->bit_rate       = bitrate;
  _codecCtx->sample_rate    = sampleRate;
  _codecCtx->channels       = nChannels;
  _codecCtx->channel_layout = nChannels == 2 ? AV_CH_LAYOUT_STEREO : AV_CH_LAYOUT_MONO;
  //_codecCtx->extradata_size = FF_MIN_BUFFER_SIZE + FF_INPUT_BUFFER_PADDING_SIZE; // for flac
  //_codecCtx->extradata = (uint8_t*)av_malloc(_codecCtx->extradata_size); //streaminfo;

  // Find encoder
  av_log_set_level(AV_LOG_VERBOSE);
  AVCodec* audioCodec = avcodec_find_encoder(_codecCtx->codec_id);
  if (!audioCodec) throw EssentiaException("Codec for ", format, " files not found or not supported");

  /*
  // TODO: add support for FLT
  if (_codecCtx->codec_id == AV_CODEC_ID_VORBIS) {
    // AV_CODEC_ID_AAC, AV_CODEC_ID_AC3 codecs requrire FLT as well 
    // Use FLT format instead of S16
    // see: https://developer.blender.org/file/data/zj4aupq36lxt3s2qskl3/PHID-FILE-66vxjiw26l4yb4nc36sn/blender_sample_not_supported.diff
    _codecCtx->sample_fmt = AV_SAMPLE_FMT_FLT;
  }
  */

  if (audioCodec->sample_fmts) {
    // check if the specified sample format is supported by the codec
    const enum AVSampleFormat* p = audioCodec->sample_fmts;
 
    while (*p != -1) {
      if (*p == _codecCtx->sample_fmt) break;
      p++;
    }
    if (*p == -1) {
      // The specified sample format is not supported
      // Default to a format supported by the coded?
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

  // Minimum size for compressed (i.e. not pcm) files is FF_MIN_BUFFER_SIZE. However flac
  // uses very big sizes which force us to set output buffer size to 4*FF_MIN_BUFFER_SIZE
  // For PCM format outputBufsize can be set to sthg smaller than FF_MIN_BUFFER_SIZE, only
  // take into account that when encoding the amount of data read from the input buffer is
  // buf_size * input_sample_size / output_sample_size.

  switch (_codecCtx->codec_id) {
    case AV_CODEC_ID_PCM_S16LE:
    case AV_CODEC_ID_PCM_S16BE:
    case AV_CODEC_ID_PCM_U16LE:
    case AV_CODEC_ID_PCM_U16BE:
      // PCM codecs do not provide frame size in samples
      _inputBufSize = 4096;
      _outputBufSize = 4096;
      _codecCtx->frame_size = _inputBufSize/_codecCtx->channels/SAMPLE_SIZE_RATIO;
      break;

    case AV_CODEC_ID_FLAC:
    case AV_CODEC_ID_VORBIS:
      _inputBufSize = av_samples_get_buffer_size(NULL, 
                                                _codecCtx->channels, 
                                                _codecCtx->frame_size, 
                                                _codecCtx->sample_fmt, 0);
      _outputBufSize = 65536;
      break;

    default:
      if (_codecCtx->frame_size <= 1) {
        // we could use these defaults, but it might not be desired
        //_inputBufSize = FF_MIN_BUFFER_SIZE;
        //_outputBufSize = FF_MIN_BUFFER_SIZE;
        //dataSize = 1;

        // so throw an exception instead
        throw EssentiaException("Do not know how to encode given format: ", format);
      }
      //_inputBufSize = _codecCtx->frame_size*_codecCtx->channels*SAMPLE_SIZE_RATIO;
      _inputBufSize = av_samples_get_buffer_size(NULL, 
                                                _codecCtx->channels, 
                                                _codecCtx->frame_size, 
                                                _codecCtx->sample_fmt, 0);
      _outputBufSize = FF_MIN_BUFFER_SIZE;
  }

  // allocate audio buffers
  _inputBuffer = (int16_t*)av_malloc(_inputBufSize);
  _outputBuffer = (uint8_t*)av_malloc(_outputBufSize);

  strncpy(_muxCtx->filename, _filename.c_str(), sizeof(_muxCtx->filename));

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

  // close output file
  if (_isOpen) {
    writeEOF();

    // Write trailer to the end of the file
    av_write_trailer(_muxCtx);

    #if LIBAVFORMAT_VERSION_INT < ((52<<16)+(0<<8)+0)
    url_fclose(&_muxCtx->pb);
    #else
    avio_close(_muxCtx->pb);
    #endif
  }

  avcodec_close(_avStream->codec);

  av_freep(&_inputBuffer);
  av_freep(&_outputBuffer);
  av_freep(&_avStream->codec);
  av_freep(&_avStream);
  av_freep(&_muxCtx); // also must be av_free, not av_freep

  _muxCtx = 0;
  _avStream = 0;
  _codecCtx = 0;
  _inputBuffer = 0;
  _outputBuffer = 0;

  _isOpen = false;
}


void AudioContext::write(const vector<StereoSample>& stereoData) {
  if (_codecCtx->channels != 2) {
    throw EssentiaException("Trying to write stereo audio data to an audio file with ", _codecCtx->channels, " channels");
  }

  int dsize = (int)stereoData.size();
  
  if (dsize > _codecCtx->frame_size) {
    // Do a double-check here although this should never happen because AudioWriter 
    // sets up correct buffer sizes in accordance to what AudioContext:create() returns
    ostringstream msg;
    msg << "Audio frame size " << _codecCtx->frame_size << 
           " is not sufficent to store " << dsize << " samples";
    throw EssentiaException(msg);
  }

  for (int i=0; i<dsize; ++i) {
    _inputBuffer[2*i] = scale(stereoData[i].left());
    _inputBuffer[2*i+1] = scale(stereoData[i].right());
  }

  encodePacket(dsize);
}


void AudioContext::write(const vector<AudioSample>& monoData) {
  if (_codecCtx->channels != 1) {
    throw EssentiaException("Trying to write mono audio data to an audio file with ", _codecCtx->channels, " channels");
  }

  int dsize = (int)monoData.size();
  if (dsize > _codecCtx->frame_size) {
    // The same as for stereoData version of write()
    ostringstream msg;
    msg << "Audio frame size " << _codecCtx->frame_size << 
           " is not sufficent to store " << dsize << " samples";
    throw EssentiaException(msg);
  }

  for (int i=0; i<dsize; ++i) _inputBuffer[i] = scale(monoData[i]);
  encodePacket(dsize);
}


void AudioContext::encodePacket(int size) {
  int tmp_fs = _codecCtx->frame_size;
  if (size < _codecCtx->frame_size) {
    _codecCtx->frame_size = size;
  }

  AVPacket packet;
  av_init_packet(&packet);
  // Set the packet data and size so that it is recognized as being empty.
  packet.data = NULL;
  packet.size = 0;

  AVFrame *frame;
  frame = av_frame_alloc();  
  if (!frame) {
    throw EssentiaException("Error allocating audio frame");
  }

  frame->nb_samples = _codecCtx->frame_size;
  frame->format = _codecCtx->sample_fmt;
  frame->channel_layout = _codecCtx->channel_layout;

  // Actual number of bytes used to store the input samples in buffer
  int frame_bytes = av_samples_get_buffer_size(NULL, _codecCtx->channels, 
                                              _codecCtx->frame_size, _codecCtx->sample_fmt, 0); 

  int result = avcodec_fill_audio_frame(frame, _codecCtx->channels, _codecCtx->sample_fmt,
                                 (const uint8_t*) _inputBuffer, frame_bytes, 0);
  if (result != 0) {
    char errstring[1204];
    av_strerror(result, errstring, sizeof(errstring));
    ostringstream msg;
    msg << "Could not setup audio frame: " << errstring;
    throw EssentiaException(msg);
  }

  
  int got_output;

  if (avcodec_encode_audio2(_codecCtx, &packet, frame, &got_output) < 0) {
     throw EssentiaException("Error while encoding audio frame");
  }

  _codecCtx->frame_size = tmp_fs;

  /*  
  cout << "\tpacket size: " << packet.size
       << "\tnum samples: " << size
       << "\tframe bytes: " << frame_bytes
       << "\toutput buf size: " << _outputBufSize
       << "\tinput buf size: " << _inputBufSize
       // << "\tduration: " << duration
       << endl;
  */

  if (got_output) { // packet is not empty, write the frame in the media file
    if (av_write_frame(_muxCtx, &packet) != 0 ) {
      throw EssentiaException("Error while writing audio frame");
    }
    av_free_packet(&packet);
  }
  
  av_frame_free(&frame);
}

void AudioContext::writeEOF() { 
  AVPacket packet;
  av_init_packet(&packet);
  // Set the packet data and size so that it is recognized as being empty.
  packet.data = NULL;
  packet.size = 0;

  for (int got_output = 1; got_output;) {
    if (avcodec_encode_audio2(_codecCtx, &packet, NULL, &got_output) < 0) {
      throw EssentiaException("Error while encoding audio frame");
    }
    if (got_output) {
      if (av_write_frame(_muxCtx, &packet) != 0 ) {
        throw EssentiaException("Error while writing delayed audio frame");
      }
      av_free_packet(&packet);
    }
    else break;
  }
}

int16_t AudioContext::scale(AudioSample value) {
  int32_t result = 0;
  if( value > 0 ) result = (int32_t)(value*32767+0.5);
  else if( value < 0 ) result = (int32_t)(value*32767 - 0.5);
  if (result > 32767) return 32767; 
  if (result < -32768) return -32768;
  return result;
}
