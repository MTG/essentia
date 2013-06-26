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

#include "audioloader.h"
#include "algorithmfactory.h"


using namespace std;

namespace essentia {
namespace streaming {

const char* AudioLoader::name = "AudioLoader";
const char* AudioLoader::description = DOC("This algorithm loads the single audio stream contained in the given audio or video file, as well as the samplerate and the number of channels. Supported formats are all those supported by the ffmpeg library, which is, virtually everything.\n"
"\n"
"This algorithm will throw an exception if it hasn't been properly configured which is normally due to not specifying a valid filename.\n"
"Note: ogg files are decoded in reverse phase, due to a (possible) bug in the ffmpeg library.\n"
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


AudioLoader::~AudioLoader() {
    closeAudioFile();

    av_freep(&_buffer);

#if LIBAVCODEC_VERSION_INT >= AVCODEC_AUDIO_DECODE4
    av_freep(&_decodedFrame);
#endif

#if !HAVE_SWRESAMPLE
    av_freep(&_buff1);
    av_freep(&_buff2);
    if (_audioConvert) {
        av_audio_convert_free(_audioConvert);
        _audioConvert = NULL;
    }
#endif
}

void AudioLoader::configure() {
    // set ffmpeg to be silent by default, so we don't have these annoying
    // "invalid new backstep" messages anymore, when everything is actually fine
    av_log_set_level(AV_LOG_QUIET);
    reset();
}



void AudioLoader::openAudioFile(const string& filename) {

    // Open file
    if (avformat_open_input(&_demuxCtx, filename.c_str(), NULL, NULL) != 0) {
        throw EssentiaException("AudioLoader: Could not open file \"", filename, "\"");
    }

    // Retrieve stream information
    if (avformat_find_stream_info(_demuxCtx, NULL) < 0) {
        avformat_close_input(&_demuxCtx);
        _demuxCtx = 0;
        throw EssentiaException("AudioLoader: Could not find stream information");
    }

    // Dump information about file onto standard error
    //dump_format(_demuxCtx, 0, filename.c_str(), 0);

    // Check that we have only 1 audio stream in the file
    int nAudioStreams = 0;
    for (int i=0; i<(int)_demuxCtx->nb_streams; i++) {
        if (_demuxCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_AUDIO) {
            _streamIdx = i;
            nAudioStreams++;
        }
    }
    if (nAudioStreams != 1) {
        throw EssentiaException("AudioLoader ERROR: found ", nAudioStreams, " streams in the file, expecting only one audio stream");
    }

    // Load corresponding audio codec
    _audioCtx = _demuxCtx->streams[_streamIdx]->codec;

    _audioCodec = avcodec_find_decoder(_audioCtx->codec_id);

    if (!_audioCodec) {
        throw EssentiaException("AudioLoader: Unsupported codec!");
    }

    if (avcodec_open2(_audioCtx, _audioCodec, NULL) < 0) {
        throw EssentiaException("AudioLoader: Unable to instantiate codec...");
    }

    if (_audioCtx->sample_fmt != AV_SAMPLE_FMT_S16) {

#if HAVE_SWRESAMPLE

        E_DEBUG(EAlgorithm, "AudioLoader: using sample format conversion from "
                "libswresample for file: " << parameter("filename").toString());

        // No samplerate conversion yet, only format
        int64_t layout = av_get_default_channel_layout(_audioCtx->channels);

        _convertCtx = swr_alloc_set_opts(_convertCtx,
                                         layout, AV_SAMPLE_FMT_S16,     _audioCtx->sample_rate,
                                         layout, _audioCtx->sample_fmt, _audioCtx->sample_rate,
                                         0, NULL);

        if (swr_init(_convertCtx) < 0) {
            throw EssentiaException("Could not initialize swresample context");
        }

        /*
        const char* fmt = 0;
        get_format_from_sample_fmt(&fmt, _audioCtx->sample_fmt);
        E_DEBUG(EAlgorithm, "AudioLoader: converting from " << (fmt ? fmt : "unknown") << " to S16");
        */

#else

        E_DEBUG(EAlgorithm, "AudioLoader: using sample format conversion from "
                "deprecated audioconvert for file: " << parameter("filename").toString());

        _audioConvert = av_audio_convert_alloc(AV_SAMPLE_FMT_S16, 1, _audioCtx->sample_fmt, 1, NULL, 0);

        // reserve some more space
        _buff1 = (int16_t*)av_malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3);
        _buff2 = (int16_t*)av_malloc(AVCODEC_MAX_AUDIO_FRAME_SIZE * 3);

#endif

    }
    else {
        E_DEBUG(EAlgorithm, "AudioLoader: no sample format conversion, direct copy "
                "for file: " << parameter("filename").toString());
    }

    av_init_packet(&_packet);

#if LIBAVCODEC_VERSION_INT >= AVCODEC_AUDIO_DECODE4
    _decodedFrame = avcodec_alloc_frame();
    if (!_decodedFrame) {
        throw EssentiaException("Could not allocate audio frame");
    }
#endif

}


void AudioLoader::closeAudioFile() {
    if (!_demuxCtx) {
        return;
    }

#if HAVE_SWRESAMPLE
    if (_convertCtx) swr_free(&_convertCtx);
#endif

    // Close the codec
    avcodec_close(_audioCtx);

    // Close the audio file
    avformat_close_input(&_demuxCtx);

    _demuxCtx = 0;
    _audioCtx = 0;
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



AlgorithmStatus AudioLoader::process() {
    if (!parameter("filename").isConfigured()) {
        throw EssentiaException("AudioLoader: Trying to call process() on an AudioLoader algo which hasn't been correctly configured.");
    }

  // read frames until we got a good one
    do {
        int result = av_read_frame(_demuxCtx, &_packet);

        if (result != 0) {
            shouldStop(true);
            flushPacket();
            closeAudioFile();
            return FINISHED;
        }
    } while (_packet.stream_index != _streamIdx);

    decodePacket();
    copyFFmpegOutput();

    return OK;
}


int AudioLoader::decode_audio_frame(AVCodecContext* audioCtx,
                                    int16_t* output,
                                    int* outputSize,
                                    AVPacket* packet) {


#if LIBAVCODEC_VERSION_INT < AVCODEC_51_28_0

    int len = avcodec_decode_audio(audioCtx, output, outputSize,
                                 packet->data, packet->size);

#elif LIBAVCODEC_VERSION_INT < AVCODEC_52_47_0

    int len = avcodec_decode_audio2(audioCtx, output, outputSize,
                                    packet->data, packet->size);

#elif LIBAVCODEC_VERSION_INT < AVCODEC_AUDIO_DECODE4

    int len = avcodec_decode_audio3(audioCtx, output, outputSize,
                                    packet);

#else

    int gotFrame = 0;
    avcodec_get_frame_defaults(_decodedFrame);

    int len = avcodec_decode_audio4(audioCtx, _decodedFrame, &gotFrame, packet);
    if (len < 0) throw EssentiaException("Error while decoding");

    if (gotFrame) {
        int nsamples = _decodedFrame->nb_samples;
        int inputDataSize = av_samples_get_buffer_size(NULL, audioCtx->channels, nsamples,
                                                       audioCtx->sample_fmt, 1);

#  if HAVE_SWRESAMPLE
        if (_convertCtx) {
            int outputSamples = *outputSize / (2 /*sizeof(S16)*/ * _nChannels);
            //if (outputSamples < nsamples) { cout << "OOPS!!" << endl; }

            if (swr_convert(_convertCtx,
                            (uint8_t**) &output, outputSamples,
                            (const uint8_t**)_decodedFrame->data, nsamples) < 0) {
                ostringstream msg;
                msg << "AudioLoader: Error converting"
                    << " from " << av_get_sample_fmt_name(_audioCtx->sample_fmt)
                    << " to "   << av_get_sample_fmt_name(AV_SAMPLE_FMT_S16);
                throw EssentiaException(msg);
            }
            *outputSize = nsamples * (2 /*sizeof(S16)*/ * _nChannels);
        }
        else {
            // no conversion needed, make a direct copy
            // copy and convert data from our frame to our output audio buffer
            //E_WARNING("Should use swresample always!");
            memcpy(output, _decodedFrame->data[0], inputDataSize);
            *outputSize = inputDataSize;
        }
#  else
        // direct copy, we do the sample format conversion later if needed
        memcpy(output, _decodedFrame->data[0], inputDataSize);
        *outputSize = inputDataSize;
#  endif

    }
    else {
        E_DEBUG(EAlgorithm, "AudioLoader: tried to decode packet but didn't get any frame...");
        *outputSize = 0;
    }

#endif

    return len;
}


void AudioLoader::flushPacket() {
    AVPacket empty;
    do {
        _dataSize = FFMPEG_BUFFER_SIZE * sizeof(int16_t);
        empty.data = 0;
        empty.size = 0;

        decode_audio_frame(_audioCtx, _buffer, &_dataSize, &empty);
        copyFFmpegOutput();

    } while (_dataSize > 0);
}


/**
 * Gets the AVPacket stored in _packet, and decodes all the samples it can from it,
 * putting them in _buffer, the total number of bytes written begin stored in _dataSize.
 */
int AudioLoader::decodePacket() {
  /*
  cout << "-----------------------------------------------------" << endl;
  cout << "decoding packet of " << bufSize << " bytes" << endl;
  cout << "pts: " << _packet.pts << " - dts: " << _packet.dts << endl; //" - pos: " << pkt->pos <<  endl;
  cout << "flags: " << _packet.flags << endl;
  cout << "duration: " << _packet.duration << endl;
  */

    int len = 0;

    // make a local copy of the packet as we might modify it if it contains
    // more than one frame, but we will have to call av_free_packet on the
    // original packet in the end
    AVPacket packet;
    packet.data = _packet.data;
    packet.size = _packet.size;

    // buff is an offset in our output buffer, it points to where we should start
    // writing the next decoded samples
    int16_t* buff = _buffer;

#if !HAVE_SWRESAMPLE
    if (_audioConvert) { buff = _buff1; }
#endif

    int totalBytesWritten = 0;

    while (packet.size > 0) { // while we still have some bytes to read from the input packet
        buff += totalBytesWritten;

        // _dataSize gets the size of this remaining buffer, in bytes
        _dataSize = FFMPEG_BUFFER_SIZE*sizeof(int16_t) - totalBytesWritten;

        //cout << "_dataSize: " << _dataSize << " " << AVCODEC_MAX_AUDIO_FRAME_SIZE << endl;
        // just to make sure that we dimensioned our buffers correctly. If this assert fails
        // we're gonna have to save audio frames contained in a packet in a temp place, then
        // join them later
        if (_dataSize < AVCODEC_MAX_AUDIO_FRAME_SIZE) {
            // asserts can be turned off, exceptions can't...
            throw EssentiaException("INTERNAL ERROR in Audioloader: the output buffer is too small; please report this bug with the corresponding audio file.");
        }

        // _dataSize  input = number of bytes available for write in buff
        //           output = number of bytes actually written (actual: S16 data)
        len = decode_audio_frame(_audioCtx, buff, &_dataSize, &packet);

        if (len < 0) {
            // only print error msg when file is not an mp3, because mp3 streams can have tag
            // frames (id3v2?) which libavcodec tries to read as audio anyway, and we don't want
            // to print an error message for that...
            if (_audioCtx->codec_id != CODEC_ID_MP3) {
                E_WARNING("AudioLoader: error while decoding, skipping frame");
                _packet.size = 0;
            }
            break;
        }

        // advance inside input buffer
        packet.data += len;
        packet.size -= len;

        if (_dataSize <= 0) {
            // No data yet, get more frames
            //cout << "no data yet, get more frames" << endl;
            continue;
        }

#if !HAVE_SWRESAMPLE
        if (_audioConvert) {
            // this assumes that all audio is interleaved in the first channel
            // it works as we're only doing sample format conversion, but we
            // should be very careful
            const void* ibuf[6] = { buff };
                  void* obuf[6] = { _buff2 };
            int istride[6]      = { av_get_bytes_per_sample(_audioCtx->sample_fmt) };
            int ostride[6]      = { av_get_bytes_per_sample(AV_SAMPLE_FMT_S16)     };
            int totalsamples    = _dataSize / istride[0]; // == num_samp_per_channel * num_channels

            if (av_audio_convert(_audioConvert, obuf, ostride, ibuf, istride, totalsamples) < 0) {
                ostringstream msg;
                msg << "AudioLoader: Error converting "
                    << " from " << avcodec_get_sample_fmt_name(_audioCtx->sample_fmt)
                    << " to "   << avcodec_get_sample_fmt_name(SAMPLE_FMT_S16);
                throw EssentiaException(msg);
                break;
            }

            // when entering the current block, dataSize contained the size in bytes
            // that the audio was taking in its native format. Now it needs to be set
            // to the size of the audio we're returning, after conversion
            _dataSize = totalsamples * av_get_bytes_per_sample(AV_SAMPLE_FMT_S16);
        }
#endif

        totalBytesWritten += _dataSize;

        if (packet.size > 0) {
            // more than 1 frame in a packet, happens a lot with flac for instance...
            E_DEBUG(EAlgorithm, "AudioLoader: more than 1 frame in packet, keep on reading from this packet until exhausted...");
        }
    }

    // now _dataSize should contain the total number of bytes written
    _dataSize = totalBytesWritten;

    av_free_packet(&_packet);

    return len;
}


inline Real scale(int16_t value) {
    return value / (Real)32767;
}

void AudioLoader::copyFFmpegOutput() {
    int nsamples  = _dataSize / 2 / _nChannels;

    // acquire necessary data
    bool ok = _audio.acquire(nsamples);
    if (!ok) {
        throw EssentiaException("AudioLoader: could not acquire output for audio");
    }

    vector<StereoSample>& audio = *((vector<StereoSample>*)_audio.getTokens());

    // FIXME: use libswresample
    if (_nChannels == 1) {
        for (int i=0; i<nsamples; i++) {
            audio[i].left() = scale(_buffer[i]);
        }
    }
    else { // _nChannels == 2
        for (int i=0; i<nsamples; i++) {
            audio[i].left() = scale(_buffer[2*i]);
            audio[i].right() = scale(_buffer[2*i+1]);
        }
    }

    // release data
    _audio.release(nsamples);

}

void AudioLoader::reset() {
    Algorithm::reset();

    if (!parameter("filename").isConfigured()) return;

    string filename = parameter("filename").toString();

    closeAudioFile();
    openAudioFile(filename);

    pushChannelsSampleRateInfo(_audioCtx->channels, _audioCtx->sample_rate);
}

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

const char* AudioLoader::name = "AudioLoader";
const char* AudioLoader::description = DOC("Given an audio file this algorithm loads an audio file and outputs the raw signal data, the samplerate and the number of channels. Supported formats are: wav, aiff, flac (not supported on Windows), ogg and mp3.\n"
"\n"
"This algorithm will throw an exception if it hasn't been properly configured which normally is due to not specifying a valid filename. Invalid names comprise those with extensions different than the supported  formats and non existent files.\n"
"Note: ogg files are decoded in reverse phase, due to be using ffmpeg library.\n"
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
    _srStorage = new streaming::VectorOutput<Real>(&_sampleRateStorage);
    _cStorage = new streaming::VectorOutput<int>(&_channelsStorage);

    _loader->output("audio")           >>  _audioStorage->input("data");
    _loader->output("sampleRate")      >>  _srStorage->input("data");
    _loader->output("numberChannels")  >>  _cStorage->input("data");

    _network = new scheduler::Network(_loader);
}

void AudioLoader::configure() {
    _loader->configure(INHERIT("filename"));
}

void AudioLoader::compute() {
    if (!parameter("filename").isConfigured()) {
        throw EssentiaException("AudioLoader: Trying to call compute() on an "
                                "AudioLoader algo which hasn't been correctly configured.");
    }

    Real& sampleRate = _sampleRate.get();
    int& nChannels = _channels.get();
    vector<StereoSample>& audio = _audio.get();

    _audioStorage->setVector(&audio);

    // FIXME:
    // _audio.reserve(sth_meaningful);

    _network->run();

    sampleRate = _sampleRateStorage[0];
    nChannels = _channelsStorage[0];
    // reset, so it is ready to load audio again
    reset();
}

void AudioLoader::reset() {
    _network->reset();
}

} // namespace standard
} // namespace essentia
