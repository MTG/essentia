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
  : _isOpen(false), _avStream(0), _demuxCtx(0), _codecCtx(0),
    _outputBufSize(0), _inputBufSize(0), _inputBuffer(0), _outputBuffer(0) {
  //av_log_set_level(AV_LOG_VERBOSE);
  av_log_set_level(AV_LOG_QUIET);
  // Register all formats and codecs
  av_register_all(); // this should be done once only..
}


int AudioContext::create(const std::string& filename,
                         const std::string& format,
                         int nChannels, int sampleRate, int bitrate) {

  if (_demuxCtx != 0) close();

  _filename = filename;

  AVOutputFormat* av_output_format = av_guess_format(format.c_str(), 0, 0);
  if (!av_output_format) {
    throw EssentiaException("Could not find a suitable output format for \"", filename, "\"");
  }
  if (format != av_output_format->name) {
    cout << "WARNING: Essentia is using a different format than the one supplied. Format used is " << av_output_format->name << endl;
  }

  _demuxCtx = avformat_alloc_context();
  if (!_demuxCtx) throw EssentiaException("Could not allocate the format context");

  _demuxCtx->oformat = av_output_format;

  // Create audio stream
  _avStream = avformat_new_stream(_demuxCtx, NULL);
  if (!_avStream) throw EssentiaException("Could not allocate stream");
  _avStream->id = 1; // necessary? found here: http://sgros.blogspot.com.es/2013/01/deprecated-functions-in-ffmpeg-library.html


  // Load corresponding codec and set it up:
  _codecCtx                 = _avStream->codec;
  _codecCtx->codec_id       = _demuxCtx->oformat->audio_codec;
  _codecCtx->codec_type     = AVMEDIA_TYPE_AUDIO;
  _codecCtx->sample_fmt     = AV_SAMPLE_FMT_S16;
  _codecCtx->bit_rate       = bitrate;
  _codecCtx->sample_rate    = sampleRate;
  _codecCtx->channels       = nChannels;
  _codecCtx->extradata_size = FF_MIN_BUFFER_SIZE + FF_INPUT_BUFFER_PADDING_SIZE; // for flac
  _codecCtx->extradata = (uint8_t*)av_malloc(_codecCtx->extradata_size);//streaminfo;

  // Set output parameters
  // FIXME: this has been commented for compatibility with newer ffmpeg api but no
  //        replacement has been found. Is one even necessary?
  //if (av_set_parameters(_demuxCtx, NULL) < 0) throw EssentiaException("Invalid output format parameters");

  // Find encoder
  //av_log_set_level(AV_LOG_VERBOSE);
  AVCodec* audioCodec = avcodec_find_encoder(_codecCtx->codec_id);
  if (!audioCodec) throw EssentiaException("Codec for ", format, " files not found or not supported");

  // Open codec and store it in _codecCtx. (OLD TODO: avcodec_open: This function is not thread safe!! see libavcodec)
  if (avcodec_open2(_codecCtx, audioCodec, NULL) < 0) {
    throw EssentiaException("AudioWriter: Could not open codec for ", format, " files");
  }

  // Minimum size for compressed (i.e. not pcm) files is FF_MIN_BUFFER_SIZE. However flac
  // uses very big sizes which force us to set output buffer size to 4*FF_MIN_BUFFER_SIZE
  // For PCM format outputBufsize can be set to sthg smaller than FF_MIN_BUFFER_SIZE, only
  // take into account that when encoding the amount of data read from the input buffer is
  // buf_size * input_sample_size / output_sample_size.

  _isFlac = false;
  int dataSize = 1;

  switch (_codecCtx->codec_id) {
    case CODEC_ID_PCM_S16LE:
    case CODEC_ID_PCM_S16BE:
    case CODEC_ID_PCM_U16LE:
    case CODEC_ID_PCM_U16BE:
      _inputBufSize = 4096;
      _outputBufSize = 4096;
      dataSize = _inputBufSize/_codecCtx->channels/SAMPLE_SIZE_RATIO;
      break;

    case CODEC_ID_FLAC:
    case CODEC_ID_VORBIS:
      _isFlac = true;
      _inputBufSize = _codecCtx->frame_size*_codecCtx->channels*SAMPLE_SIZE_RATIO;
      _outputBufSize = 65536;
      dataSize = _codecCtx->frame_size;
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

      _inputBufSize = _codecCtx->frame_size*_codecCtx->channels*SAMPLE_SIZE_RATIO;
      _outputBufSize = FF_MIN_BUFFER_SIZE;
      dataSize = _codecCtx->frame_size;
  }

  /////// HACK USED IN FFMPEG EXAMPLES ///////
//  if (_codecCtx->frame_size <= 1) { // pcm format only
//    _inputBufSize = 4096;
//    _outputBufSize = 4096;
//    switch(_codecCtx->codec_id) {
//      case CODEC_ID_PCM_S16LE:
//      case CODEC_ID_PCM_S16BE:
//      case CODEC_ID_PCM_U16LE:
//      case CODEC_ID_PCM_U16BE:
//        dataSize = _inputBufSize/_codecCtx->channels/SAMPLE_SIZE_RATIO;
//        break;
//      default:
//        break;
//    }
//  }
//  else {
//    switch(_codecCtx->codec_id) {
//      case CODEC_ID_FLAC:
//      case CODEC_ID_VORBIS:
//        _outputBufSize = 65536;
//        _isFlac = true;
//        break;
//      default:
//        break;
//    }
//    _inputBufSize = _codecCtx->frame_size*_codecCtx->channels*SAMPLE_SIZE_RATIO;
//    dataSize = _codecCtx->frame_size;
//  }

  // FF_INPUT_BUFFER_PADDING_SIZE is needed for some architectures
  _inputBufSize += FF_INPUT_BUFFER_PADDING_SIZE;

  // allocate audio buffers
  _inputBuffer = (int16_t*)av_malloc(_inputBufSize);
  _outputBuffer = (uint8_t*)av_malloc(_outputBufSize);

  // dump info about the encoder:
  //av_log_set_level(AV_LOG_VERBOSE);
  //dump_format(_demuxCtx, 0, filename.c_str(), 1);

  strncpy(_demuxCtx->filename, _filename.c_str(), sizeof(_demuxCtx->filename));

  return dataSize;
}


void AudioContext::open() {
  if (_isOpen) return;

  if (!_demuxCtx) throw EssentiaException("Trying to open an audio file that has not been created yet or has been closed");

  // Open output file

  if (avio_open(&_demuxCtx->pb, _filename.c_str(), AVIO_FLAG_WRITE) < 0) {
    throw EssentiaException("Could not open \"", _filename, "\"");
  }

  avformat_write_header(_demuxCtx, /* AVDictionary **options */ NULL);


  _isOpen = true;
}


void AudioContext::close() {
  if (!_demuxCtx) return;

  // close output file
  if (_isOpen) {
    writeEOF();

    // Write trailer to the end of the file
    av_write_trailer(_demuxCtx);

    #if LIBAVFORMAT_VERSION_INT < ((52<<16)+(0<<8)+0)
    url_fclose(&_demuxCtx->pb);
    #else
    avio_close(_demuxCtx->pb);
    #endif
  }

  avcodec_close(_avStream->codec);

  av_freep(&_inputBuffer);
  av_freep(&_outputBuffer);
  av_freep(&_avStream->codec);
  av_freep(&_avStream);
  av_freep(&_demuxCtx); // also must be av_free, not av_freep

  _demuxCtx = 0;
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
  for (int i=0; i<dsize; ++i) _inputBuffer[i] = scale(monoData[i]);

  encodePacket(dsize);
}

void AudioContext::encodePacket(int size) {
  int tmp_fs = _codecCtx->frame_size;
  if (size < _codecCtx->frame_size) {
    _codecCtx->frame_size = size;
  }

  int frame_bytes = size*SAMPLE_SIZE_RATIO*_codecCtx->channels;
  if (_isFlac) frame_bytes = 65536; // not mandatory on some platforms (e.g. darwin)

  AVPacket packet;
  av_init_packet(&packet);

  //Real duration = (double)_avStream->pts.val * _avStream->time_base.num / _avStream->time_base.den;
  packet.size = avcodec_encode_audio(_codecCtx, _outputBuffer, frame_bytes, (short*)_inputBuffer);
  _codecCtx->frame_size = tmp_fs;

  /*
  cout << "\tpacket size: " << packet.size
       << "\tnum samples: " << size
       << "\tframe bytes: " << frame_bytes
       << "\toutput buf size: " << _outputBufSize
       << "\tinput buf size: " << _inputBufSize
       << "\tduration: " << duration
       << endl;
  */

  if (packet.size < 0) throw EssentiaException("Error while encoding audio frame");

  if (_codecCtx->coded_frame->pts != (int)AV_NOPTS_VALUE) {
    packet.pts = av_rescale_q(_codecCtx->coded_frame->pts, _codecCtx->time_base, _avStream->time_base);
  }

  packet.flags |= AV_PKT_FLAG_KEY;
  packet.stream_index = _avStream->index;
  packet.data = _outputBuffer;

  // write the frame in the media file
  if (av_interleaved_write_frame(_demuxCtx, &packet) != 0 ) {
    throw EssentiaException("Error while writing audio frame");
  }
}

void AudioContext::writeEOF() {
  if (_codecCtx->frame_size <= 1) return; // pcm
  // the size could be shrinked to 34 (mininum for flac), probably
  int frame_bytes = 128;
  if (_isFlac) frame_bytes = 65536;

  _codecCtx->extradata = (uint8_t*)av_malloc(_codecCtx->extradata_size);
  int size = 0;

  while(true) {
    AVPacket packet;
    av_init_packet(&packet);
    if (!_isFlac) {
      int tmp_fs = _codecCtx->frame_size;
      _codecCtx->frame_size = 0;

      size = avcodec_encode_audio(_codecCtx, _outputBuffer, frame_bytes, (short*)_inputBuffer); // outputs remaining samples
      _codecCtx->frame_size = tmp_fs;
    }
    if (size <=0) {
      size = avcodec_encode_audio(_codecCtx, _outputBuffer, frame_bytes, NULL); // sets EOF
    }
    if (size == 0) break;
    if (size < 0) {
      throw EssentiaException("Error encoding last frames");
    }
    packet.size = size;
    packet.flags |= AV_PKT_FLAG_KEY;
    packet.stream_index = _avStream->index;
    packet.data = _outputBuffer;

    if (av_interleaved_write_frame(_demuxCtx, &packet) != 0 ) {
      throw EssentiaException("Error while writing last frames");
    }
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
