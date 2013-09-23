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

#ifndef ESSENTIA_FFMPEGAPI_H
#define ESSENTIA_FFMPEGAPI_H


// TODO: as soon as debian sorts its multimedia mess and include ffmpeg again,
//       this can be removed along with all the outdated code related to it
#ifndef HAVE_SWRESAMPLE
#define HAVE_SWRESAMPLE 0
#endif

// TODO Long-term: get rid of audioconvert.* and switch to using official libavresample API.
/* Current libavcodec-dev package is missing /usr/include/libavcodec/audioconvert.h,
   therefore we have a copy of it in the source. Audioconvert is not a public header
   in either libav or ffmpeg. What is supposed to be used for this functionality is
   libavresample or libswresample.

   The problem with that is that audioconvert.h is not part of the public
   API. Moreover, most of the APIs have already been removed in current
   libav/master in favor of the newly introduced libavresample library.
   Therefore, I do not think it would be a good idea to ship this header.

   The proper long-term solution is to port handbrake to 'libavresample'
   (it is in experimental and is not going to be included in wheezy). As short-term workaround,
   the audioconvert.h and audioconvert.c are copied.
*/


extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

// libav* versions for deprecated functions taken from (among other sources):
// https://github.com/tuttleofx/TuttleOFX/pull/23#issuecomment-6350715
#define AVCODEC_51_28_0   AV_VERSION_INT(51, 28, 0)
#define AVCODEC_52_47_0   AV_VERSION_INT(52, 47, 0)
#define AVCODEC_53_0_0    AV_VERSION_INT(53,  0, 0)
#define AVCODEC_53_8_0    AV_VERSION_INT(53,  8, 0)
#define AVCODEC_53_25_0   AV_VERSION_INT(53, 25, 0)

#define AVFORMAT_52_26_0  AV_VERSION_INT(52, 26, 0)
#define AVFORMAT_53_6_0   AV_VERSION_INT(53,  6, 0)
#define AVFORMAT_53_17_0  AV_VERSION_INT(53, 17, 0)

// useful aliases
#define AVCODEC_AUDIO_DECODE4 AVCODEC_53_25_0


// deprecated functions equivalences
#if LIBAVCODEC_VERSION_INT < AVCODEC_53_0_0
#   define AVMEDIA_TYPE_AUDIO CODEC_TYPE_AUDIO
#endif

#if LIBAVCODEC_VERSION_INT < AVCODEC_53_8_0
#   define avcodec_open2(a, c, o) avcodec_open(a, c)
#endif

#if LIBAVFORMAT_VERSION_INT < AVFORMAT_53_17_0
#   define avformat_open_input(ctx, f, x, y)  av_open_input_file(ctx, f, x, 0, y)
#   define avformat_close_input(ctx) av_close_input_file(*ctx)
#endif

#if LIBAVFORMAT_VERSION_INT < AVFORMAT_53_6_0
#   define avformat_new_stream av_new_stream
#   define avformat_set_parameters av_set_parameters
#   define avformat_find_stream_info(ctx, o) av_find_stream_info(ctx)
#endif

#if LIBAVFORMAT_VERSION_INT < AVFORMAT_52_26_0
#   define avformat_alloc_context av_alloc_format_context
#endif


// --- audioconvert

extern "C" {

#if HAVE_SWRESAMPLE
#   include <libswresample/swresample.h>
#else
#   include "audioconvert.h"
#endif

}


// --- from audiocontext

// libav API changes, get some defines to have backwards compatibility
#if (LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(52, 45, 0))
#   define av_guess_format         guess_format
#   define AVSampleFormat          SampleFormat
#   define AV_SAMPLE_FMT_S16       SAMPLE_FMT_S16
#endif

#if LIBAVCODEC_VERSION_INT < AVCODEC_53_0_0
#   define AVMediaType             CodecType
#   define AVMEDIA_TYPE_AUDIO      CODEC_TYPE_AUDIO
#   define AVMEDIA_TYPE_VIDEO      CODEC_TYPE_VIDEO
#   define AVMEDIA_TYPE_SUBTITLE   CODEC_TYPE_SUBTITLE
#   define AVMEDIA_TYPE_DATA       CODEC_TYPE_DATA
#   define AVMEDIA_TYPE_ATTACHMENT CODEC_TYPE_ATTACHMENT
#endif

#ifndef AV_PKT_FLAG_KEY
#   define AV_PKT_FLAG_KEY         PKT_FLAG_KEY
#endif

#ifndef AVIO_FLAG_WRITE
#   define AVIO_FLAG_WRITE URL_WRONLY
#   define avio_open  url_fopen
#   define avio_close url_fclose
#   define avformat_write_header(c, o) av_write_header(c)
#endif


#endif // ESSENTIA_FFMPEGAPI_H
