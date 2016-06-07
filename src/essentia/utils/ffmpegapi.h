/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/md5.h>
#include<libavresample/avresample.h>
#include<libavutil/opt.h>
}


// --- from audiocontext

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

