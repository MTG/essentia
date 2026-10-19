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

#ifdef _WIN32
#include <windows.h>
#endif

#include <fileref.h>
#include <tpropertymap.h>
#include <tag.h>

#include <algorithm>

#include "metadatareader.h"
#include "metadatautils.h"
#include "essentiautil.h"

#ifdef ESSENTIA_HAVE_LIBAVFORMAT
#include "ffmpegapi.h"
#endif


using namespace std;

string fixInvalidUTF8(const string& str) {
  // a big fat hack to try to fix invalid utf-8 characters
  // see http://www.utf8-chartable.de/
  // http://stackoverflow.com/questions/6555015/check-for-invalid-utf8
  // http://stackoverflow.com/questions/17316506/strip-invalid-utf8-from-string-in-c-c
  string fixed;
  fixed.reserve(str.size());
  unsigned char c, c2=0, c3=0, c4=0;

  for(int i=0; i<(int)str.size(); i++) {
    c = (unsigned char)str[i];

    if (c < 127) { // normal ascii
      if (c==9 || c==10 || c==13 || c >= 32) { // normal char or \t \n \r
        fixed += c;
      }
    }
    else if (c < 160) { // control character
      if (c2 == 128) { // fix microsoft mess, add euro
        fixed += 226;
        fixed += 130;
        fixed += 172;
      }
      if (c2 == 133) { // fix IBM mess, add NEL = \n\r
        fixed += 10;
        fixed += 13;
      }
    }
    else if (c<192) { // invalid for utf8, converting ascii
      fixed += (unsigned char)194;
      fixed += c;
    }
    else if (c<194) { // invalid for utf8, converting ascii
      fixed += (unsigned char)195;
      fixed += c-64;
    }
    else if(c < 224) { // possibly two-byte utf8
      c2=(unsigned char)str[i+1];
      if (c2>127 && c2<192) { // valid two-byte utf8
        if (c==194 && c2<160) { // control char, skipping
          ;
        }
        else {
          fixed += c;
          fixed += c2;
        }
        i++;
      }
      else { // invalid utf8, converting ascii
        fixed += (unsigned char)195;
        fixed += c-64;
      }
    } else if (c < 240) { // possibly three-byte utf8
      c2=(unsigned char)str[i+1];
      c3=(unsigned char)str[i+2];
      if (c2>127 && c2<192 && c3>127 && c3<192) { // valid three-byte utf8
        fixed += c;
        fixed += c2;
        fixed += c3;
        i += 2;
      }
      else { // invalid utf8, converting ascii
        fixed += (unsigned char)195;
        fixed += c-64;
      }
    } else if (c<245) { // possibly four-byte utf8
      c2=(unsigned char)str[i+1];
      c3=(unsigned char)str[i+2];
      c4=(unsigned char)str[i+3];
      if (c2>127 && c2<192 && c3>127 && c3<192 && c4>127 && c4<192) {
        // valid four-byte utf8
        fixed += c;
        fixed += c2;
        fixed += c3;
        fixed += c4;
        i += 3;
      } else { // invalid utf8, converting ascii
        fixed += (unsigned char)195;
        fixed += c-64;
      }
    }
    else { // invalid utf8, converting ascii
      fixed += (unsigned char)195;
      fixed += c-64;
    }
  }
  return fixed;
}


bool containsControlChars(const string& str) {
  for (int i=0; i<(int)str.size(); i++) {
    int c = (unsigned char)str[i];
    if ((c >= 0x00 && c <= 0x1F &&
         c != 0x09 && c != 0x0A && c != 0x0D) || // C0 control code set minus newlines & tabs
        (c >= 0x80 && c <= 0x9F)) { // C1 control code set
      return true;
    }
  }
  return false;
}

bool isLatin1(const TagLib::String& str) {
  return str.isLatin1();
}


// Utility function to format tags so that they can be correctly parsed back
string formatString(const TagLib::StringList& strList) {
  TagLib::String str = strList.toString(";");
  if (str.isEmpty()) return "";

  string result = str.to8Bit(true);

  // heuristic to detect wrongly encoded tags (ie: twice latin-1 to utf-8, mostly)
  // we should encode everything ourselves to utf-8, but sometimes it might happen
  // that someone already did that, but told us the string was in latin-1.
  // A way to detect that is if the string contains only latin-1 chars, when
  // converting it to latin-1 it contains code chars, this probably means it was
  // previously encoded in utf-8
  if (isLatin1(str) &&
      containsControlChars(str.to8Bit(false))) {
    result = str.to8Bit(false);
  }

  // fix invalid utf-8 characters
  result = fixInvalidUTF8(result);

  return result;
}

// Populate a pool with the tags found in a TagLib::PropertyMap, optionally
// filtering them with a white-list of tag names.
static void fillTagPool(const TagLib::PropertyMap& tags,
                        bool filterMetadata,
                        const std::vector<std::string>& filterMetadataTags,
                        const std::string& tagPoolName,
                        essentia::Pool& tagPool) {
  for(TagLib::PropertyMap::ConstIterator i = tags.begin(); i != tags.end(); ++i) {
    string key = i->first.to8Bit(true);
    if (!filterMetadata || std::find(filterMetadataTags.begin(), filterMetadataTags.end(), key) != filterMetadataTags.end()) {
        // remove '.' chars which are used in Pool descriptor names as a separator
        // convert to lowercase
        std::replace(key.begin(), key.end(), '.', '_');
        std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        key = tagPoolName + "." + key;

        for(TagLib::StringList::ConstIterator str = i->second.begin(); str != i->second.end(); ++str) {
          tagPool.add(key, str->to8Bit(true));
        }
    }
  }
}


#ifdef ESSENTIA_HAVE_LIBAVFORMAT

// Merge an avformat metadata dictionary into a TagLib::PropertyMap so that
// tags found by the libavformat fallback can be processed by the same code
// that processes tags found by TagLib. Keys already present in the map are
// appended to.
static void mergeAVDictionary(const AVDictionary* dict, TagLib::PropertyMap& tags) {
  if (!dict) return;
  const AVDictionaryEntry* entry = NULL;
  while ((entry = av_dict_get(dict, "", entry, AV_DICT_IGNORE_SUFFIX)) != NULL) {
    if (!entry->key || !entry->value || !*entry->value) continue;
    string key = entry->key;
    std::transform(key.begin(), key.end(), key.begin(), ::toupper);
    // translate ffmpeg's generic tag name to the TagLib naming convention
    if (key == "TRACK") key = "TRACKNUMBER";
    TagLib::String tagKey(key, TagLib::String::UTF8);
    TagLib::String tagValue(entry->value, TagLib::String::UTF8);
    if (tags.contains(tagKey)) {
      tags[tagKey].append(tagValue);
    }
    else {
      tags.insert(tagKey, TagLib::StringList(tagValue));
    }
  }
}

// Fallback metadata reader based on libavformat, which Essentia already uses
// for audio decoding (AudioLoader). It is used when TagLib cannot parse the
// audio file at all -- e.g., Matroska (.mka/.mkv/.webm) requires TagLib >= 2.2,
// which not all builds ship with -- or when TagLib parses the file but cannot
// provide its audio properties. Returns true if libavformat could open the
// file and find an audio stream in it.
static bool avformatMetadata(const string& filename,
                             int& duration, int& bitrate,
                             int& sampleRate, int& channels,
                             TagLib::PropertyMap* tags = NULL) {
  AVFormatContext* fmtCtx = NULL;
  if (avformat_open_input(&fmtCtx, filename.c_str(), NULL, NULL) != 0) {
    return false;
  }
  if (avformat_find_stream_info(fmtCtx, NULL) < 0) {
    avformat_close_input(&fmtCtx);
    return false;
  }

  int streamIdx = av_find_best_stream(fmtCtx, AVMEDIA_TYPE_AUDIO, -1, -1, NULL, 0);
  if (streamIdx < 0) {
    avformat_close_input(&fmtCtx);
    return false;
  }

  const AVStream* stream = fmtCtx->streams[streamIdx];
  const AVCodecParameters* codecParams = stream->codecpar;

  // container duration in AV_TIME_BASE units; fall back to the stream
  // duration when the container does not provide one
  int64_t durationUs = fmtCtx->duration;
  if (durationUs <= 0 && stream->duration > 0) {
    durationUs = av_rescale_q(stream->duration, stream->time_base, AV_TIME_BASE_Q);
  }
  // round to the nearest second, as TagLib::AudioProperties::length()
  // reports integer seconds
  duration = durationUs > 0 ? (int)((durationUs + AV_TIME_BASE / 2) / AV_TIME_BASE) : 0;

  int64_t bitRate = fmtCtx->bit_rate > 0 ? fmtCtx->bit_rate : codecParams->bit_rate;
  bitrate = bitRate > 0 ? (int)(bitRate / 1000) : 0;  // kb/s, as TagLib reports it

  sampleRate = codecParams->sample_rate;
  channels = codecParams->ch_layout.nb_channels;

  if (tags) {
    mergeAVDictionary(fmtCtx->metadata, *tags);
    mergeAVDictionary(stream->metadata, *tags);
  }

  avformat_close_input(&fmtCtx);
  return true;
}

#endif // ESSENTIA_HAVE_LIBAVFORMAT


namespace essentia {
namespace standard {

const char* MetadataReader::name = "MetadataReader";
const char* MetadataReader::category = "Input/output";
const char* MetadataReader::description = DOC("This algorithm loads the metadata tags from an audio file as well as outputs its audio properties. Tags and audio properties are read with TagLib for the file types it supports (mp3, ogg, flac, mp4/m4a, and others). When TagLib cannot parse a file, two fallbacks apply in order:\n"
"  - raw PCM files (.wav, .aiff) are handled by a built-in header reader that outputs audio properties only (no tags);\n"
"  - any other file type falls back to FFmpeg's libavformat, which reads both the audio properties (duration, bitrate, sample rate, channels) and the metadata tags, when Essentia is built with FFmpeg support (e.g., Matroska containers require TagLib >= 2.2 to be read natively).\n"
"An exception is thrown if the file does not exist. If no available reader can parse the file, an exception is thrown when 'failOnError' is enabled; otherwise all outputs are left empty or zero.\n"
"If using this algorithm on Windows, you must ensure that the filename is encoded as UTF-8.\n"
"This algorithm also contains some heuristic to try to deal with encoding errors in the tags and tries to do the appropriate conversion if a problem was found (mostly twice latin1->utf8 conversion).\n"
"\n"
"MetadataReader reads all metadata tags found in audio and stores them in the pool tagPool. Standard metadata tags found in audio files include strings mentioned in [1,2]. Tag strings are case-sensitive and they are converted to lower-case when stored to the pool. It is possible to filter these tags by using 'filterMetadataTags' parameter. This parameter should specify a white-list of tag strings as they are found in the audio file (e.g., \"ARTIST\").\n"
"\n"
"References:\n"
"  [1] https://taglib.github.io/api/classTagLib_1_1PropertyMap.html#details\n\n"
"  [2] https://picard.musicbrainz.org/docs/mappings/");


void MetadataReader::configure() {
  if (parameter("filename").isConfigured()) {
    _filename = parameter("filename").toString();
  }
  _tagPoolName = parameter("tagPoolName").toString();
  _filterMetadata = parameter("filterMetadata").toBool();
  _filterMetadataTags = parameter("filterMetadataTags").toVectorString();
}

void MetadataReader::compute() {
  if (!parameter("filename").isConfigured()) {
    throw EssentiaException("MetadataReader: 'filename' parameter has not been configured");
  }

#ifdef _WIN32
  int len = MultiByteToWideChar(CP_UTF8, 0, _filename.c_str(), -1, NULL, 0);
  wchar_t *buf = (wchar_t*)malloc(sizeof(wchar_t)*len);
  memset(buf, 0, len);
  MultiByteToWideChar(CP_UTF8, 0, _filename.c_str(), -1, buf, len);
  TagLib::FileRef f(buf);
  free(buf);
#else
  TagLib::FileRef f(_filename.c_str());
#endif

  Pool tagPool;

  if (f.isNull()) {
    // TagLib cannot parse this file at all. This happens for containers that
    // the linked TagLib version does not support (e.g., Matroska requires
    // TagLib >= 2.2) as well as for raw PCM files.
    // First try some basic PCM approach, then fall back to libavformat.
    int fallbackDuration = 0;
    int fallbackSampleRate = 0;
    int fallbackChannels = 0;
    int fallbackBitrate = 0;
    bool foundMetadata = false;
    string pcmError;

    try {
      pcmMetadata(_filename, fallbackSampleRate, fallbackChannels, fallbackBitrate);
      // works only for 16bit wavs/pcm; it should output incorrect value for
      // 24bit or 32bit float files, therefore, print a warning
      E_WARNING("MetadataReader: TagLib could not get metadata for this file. The output bitrate is estimated treating the input as 16-bit PCM, and therefore may be incorrect.");
      foundMetadata = true;
    }
    catch (EssentiaException& e) {
      pcmError = e.what();
    }

    TagLib::PropertyMap fallbackTags;

#ifdef ESSENTIA_HAVE_LIBAVFORMAT
    if (!foundMetadata) {
      // not a PCM file either: read audio properties (and tags) via
      // libavformat, which supports every container Essentia can decode
      foundMetadata = avformatMetadata(_filename, fallbackDuration, fallbackBitrate,
                                       fallbackSampleRate, fallbackChannels, &fallbackTags);
    }
#endif

    if (!foundMetadata && parameter("failOnError").toBool()) {
      throw EssentiaException("MetadataReader: File does not exist or does not seem to be of a supported filetype. ", pcmError);
    }

    _title.get()   = formatString(fallbackTags["TITLE"]);
    _artist.get()  = formatString(fallbackTags["ARTIST"]);
    _album.get()   = formatString(fallbackTags["ALBUM"]);
    _comment.get() = formatString(fallbackTags["COMMENT"]);
    _genre.get()   = formatString(fallbackTags["GENRE"]);
    _track.get()   = formatString(fallbackTags["TRACKNUMBER"]);
    _date.get()    = formatString(fallbackTags["DATE"]);

    fillTagPool(fallbackTags, _filterMetadata, _filterMetadataTags, _tagPoolName, tagPool);

    _tagPool.get()  = tagPool;

    _duration.get()   = fallbackDuration;
    _bitrate.get()    = fallbackBitrate;
    _sampleRate.get() = fallbackSampleRate;
    _channels.get()   = fallbackChannels;

    return;
  }

  TagLib::PropertyMap tags = f.file()->properties();

  _title.get()   = formatString(tags["TITLE"]);
  _artist.get()  = formatString(tags["ARTIST"]);
  _album.get()   = formatString(tags["ALBUM"]);
  _comment.get() = formatString(tags["COMMENT"]);
  _genre.get()   = formatString(tags["GENRE"]);
  _track.get()   = formatString(tags["TRACKNUMBER"]);
  _date.get()    = formatString(tags["DATE"]);

  // populate tag pool
  fillTagPool(tags, _filterMetadata, _filterMetadataTags, _tagPoolName, tagPool);

  _tagPool.get()  = tagPool;

  TagLib::AudioProperties* audioProperties = f.audioProperties();
  int duration   = audioProperties ? audioProperties->length() : 0;
  int bitrate    = audioProperties ? audioProperties->bitrate() : 0;
  int sampleRate = audioProperties ? audioProperties->sampleRate() : 0;
  int channels   = audioProperties ? audioProperties->channels() : 0;

#ifdef ESSENTIA_HAVE_LIBAVFORMAT
  if (duration == 0 && sampleRate == 0) {
    // TagLib recognized the file but could not read its audio properties;
    // fill the numeric audio properties from libavformat instead
    int avDuration = 0, avBitrate = 0, avSampleRate = 0, avChannels = 0;
    if (avformatMetadata(_filename, avDuration, avBitrate, avSampleRate, avChannels)) {
      duration   = avDuration;
      bitrate    = avBitrate;
      sampleRate = avSampleRate;
      channels   = avChannels;
    }
  }
#endif

  _duration.get()   = duration;
  _bitrate.get()    = bitrate;
  _sampleRate.get() = sampleRate;
  _channels.get()   = channels;

  // fix for taglib incorrectly returning the bitrate for wave files
  string ext = toLower(_filename.substr(_filename.size()-3));
  if (ext == "wav") {
    _bitrate.get() = _bitrate.get() * 1024 / 1000;
  }
}

} // namespace standard
} // namespace essentia

namespace essentia {
namespace streaming {


const char* MetadataReader::name = standard::MetadataReader::name;
const char* MetadataReader::description = standard::MetadataReader::description;


void MetadataReader::configure() {
  _filename = parameter("filename").toString();
  _newlyConfigured = true;
}

AlgorithmStatus MetadataReader::process() {
  if (_filename == "" || !_newlyConfigured) return PASS;

  TagLib::FileRef f(_filename.c_str());

  //Pool tagPool;

  if (f.isNull()) {
    // TagLib cannot parse this file at all. This happens for containers that
    // the linked TagLib version does not support (e.g., Matroska requires
    // TagLib >= 2.2) as well as for raw PCM files.
    // First try some basic PCM approach, then fall back to libavformat.
    int fallbackDuration = 0;
    int fallbackSampleRate = 0;
    int fallbackChannels = 0;
    int fallbackBitrate = 0;
    bool foundMetadata = false;
    string pcmError;

    try {
      pcmMetadata(_filename, fallbackSampleRate, fallbackChannels, fallbackBitrate);
      foundMetadata = true;
    }
    catch (EssentiaException& e) {
      pcmError = e.what();
    }

    TagLib::PropertyMap fallbackTags;

#ifdef ESSENTIA_HAVE_LIBAVFORMAT
    if (!foundMetadata) {
      // not a PCM file either: read audio properties (and tags) via
      // libavformat, which supports every container Essentia can decode
      foundMetadata = avformatMetadata(_filename, fallbackDuration, fallbackBitrate,
                                       fallbackSampleRate, fallbackChannels, &fallbackTags);
    }
#endif

    if (!foundMetadata && parameter("failOnError").toBool()) {
      throw EssentiaException("MetadataReader: File does not exist or does not seem to be of a supported filetype. ", pcmError);
    }

    _title.push(formatString(fallbackTags["TITLE"]));
    _artist.push(formatString(fallbackTags["ARTIST"]));
    _album.push(formatString(fallbackTags["ALBUM"]));
    _comment.push(formatString(fallbackTags["COMMENT"]));
    _genre.push(formatString(fallbackTags["GENRE"]));
    _track.push(formatString(fallbackTags["TRACKNUMBER"]));
    _date.push(formatString(fallbackTags["DATE"]));
    //_tagPool.push(tagPool);
    _duration.push(fallbackDuration);
    _bitrate.push(fallbackBitrate);
    _sampleRate.push(fallbackSampleRate);
    _channels.push(fallbackChannels);
  }
  else {
    TagLib::PropertyMap tags = f.file()->properties();

    _title.push(formatString(tags["TITLE"]));
    _artist.push(formatString(tags["ARTIST"]));
    _album.push(formatString(tags["ALBUM"]));
    _comment.push(formatString(tags["COMMENT"]));
    _genre.push(formatString(tags["GENRE"]));
    _track.push(formatString(tags["TRACKNUMBER"]));
    _date.push(formatString(tags["DATE"]));


    /*
    // populate tag pool
    for(PropertyMap::Iterator it = tags.begin(); it != tags.end(); ++it) {
      for(StringList::Iterator str = it->second.begin(); str != it->second.end(); ++str) {
        tagPool.add(it->first.to8Bit(true), str->to8Bit(true));
      }
    }
    _tagPool.push(tagPool);
    */

    TagLib::AudioProperties* audioProperties = f.audioProperties();
    int duration   = audioProperties ? audioProperties->length() : 0;
    int bitrate    = audioProperties ? audioProperties->bitrate() : 0;
    int sampleRate = audioProperties ? audioProperties->sampleRate() : 0;
    int channels   = audioProperties ? audioProperties->channels() : 0;

#ifdef ESSENTIA_HAVE_LIBAVFORMAT
    if (duration == 0 && sampleRate == 0) {
      // TagLib recognized the file but could not read its audio properties;
      // fill the numeric audio properties from libavformat instead
      int avDuration = 0, avBitrate = 0, avSampleRate = 0, avChannels = 0;
      if (avformatMetadata(_filename, avDuration, avBitrate, avSampleRate, avChannels)) {
        duration   = avDuration;
        bitrate    = avBitrate;
        sampleRate = avSampleRate;
        channels   = avChannels;
      }
    }
#endif

    _duration.push(duration);

    // fix for taglib incorrectly returning the bitrate for wave files
    string ext = toLower(_filename.substr(_filename.size()-3));
    if (ext == "wav") {
      bitrate = bitrate * 1024 / 1000;
    }

    _bitrate.push((int)bitrate);
    _sampleRate.push((int)sampleRate);
    _channels.push((int)channels);
  }

  _newlyConfigured = false;
  shouldStop(true);
  return OK;
}

} // namespace streaming
} // namespace essentia
