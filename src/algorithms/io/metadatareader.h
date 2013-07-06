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

#ifndef ESSENTIA_METADATAREADER_H
#define ESSENTIA_METADATAREADER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class MetadataReader : public Algorithm {

 protected:
  // Tags
  Output<std::string> _title;
  Output<std::string> _artist;
  Output<std::string> _album;
  Output<std::string> _comment;
  Output<std::string> _genre;
  Output<int> _track;
  Output<int> _year;

  // Audio properties
  Output<int> _length;
  Output<int> _bitrate;
  Output<int> _sampleRate;
  Output<int> _channels;

  std::string _filename;

 public:
  MetadataReader() {
    declareOutput(_title, "title", "the title of the track");
    declareOutput(_artist, "artist", "the artist of the track");
    declareOutput(_album, "album", "the album on which this track appears");
    declareOutput(_comment, "comment", "the comment field stored in the tags");
    declareOutput(_genre, "genre", "the genre as stored in the tags");
    declareOutput(_track, "track", "the track number");
    declareOutput(_year, "year", "the year of publication");

    declareOutput(_length, "length", "the length of the track, in seconds");
    declareOutput(_bitrate, "bitrate", "the bitrate of the track [kb/s]");
    declareOutput(_sampleRate, "sampleRate", "the sample rate [Hz]");
    declareOutput(_channels, "channels", "the number of channels");
  }

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read the tags", "", Parameter::STRING);
    declareParameter("failOnError", "if true, the algorithm throws an exception when encountering an error (e.g. trying to open an unsupported file format), otherwise the algorithm leaves all fields blank", "{true,false}", false);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class MetadataReader : public Algorithm {

 protected:
  // Tags
  Source<std::string> _title;
  Source<std::string> _artist;
  Source<std::string> _album;
  Source<std::string> _comment;
  Source<std::string> _genre;
  Source<int> _track;
  Source<int> _year;

  // Audio properties
  Source<int> _length;
  Source<int> _bitrate;
  Source<int> _sampleRate;
  Source<int> _channels;

  std::string _filename;
  bool _newlyConfigured;

 public:
  MetadataReader() {
    declareOutput(_title, 0, "title", "the title of the track");
    declareOutput(_artist, 0, "artist", "the artist of the track");
    declareOutput(_album, 0, "album", "the album on which this track appears");
    declareOutput(_comment, 0, "comment", "the comment field stored in the tags");
    declareOutput(_genre, 0, "genre", "the genre as stored in the tags");
    declareOutput(_track, 0, "track", "the track number");
    declareOutput(_year, 0, "year", "the year of publication");

    declareOutput(_length, 0, "length", "the length of the track, in seconds");
    declareOutput(_bitrate, 0, "bitrate", "the bitrate of the track [kb/s]");
    declareOutput(_sampleRate, 0, "sampleRate", "the sample rate [Hz]");
    declareOutput(_channels, 0, "channels", "the number of channels");

  }

  void declareParameters() {
    declareParameter("filename", "the name of the file from which to read the tags", "", "");
    declareParameter("failOnError", "if true, the algorithm throws an exception when encountering an error (e.g. trying to open an unsupported file format), otherwise the algorithm leaves all fields blank", "{true,false}", false);
  }

  AlgorithmStatus process();
  void configure();
  void reset() {
    Algorithm::reset();
    _newlyConfigured = true;
  }

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_METADATAREADER_H
