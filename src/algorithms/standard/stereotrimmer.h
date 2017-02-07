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

#ifndef ESSENTIA_STEREOTRIMMER_H
#define ESSENTIA_STEREOTRIMMER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class StereoTrimmer : public Algorithm {

 private:
  Input<std::vector<StereoSample> > _input;
  Output<std::vector<StereoSample> > _output;

  long long _startIndex;
  long long _endIndex;
  bool _checkRange;


 public:
  StereoTrimmer() {
    declareInput(_input, "signal", "the input stereo signal");
    declareOutput(_output, "signal", "the trimmed stereo signal");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the input audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("startTime", "the start time of the slice you want to extract [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice you want to extract [s]", "[0,inf)", 1.0e6);
    declareParameter("checkRange", "check whether the specified time range for a slice fits the size of input signal (throw exception if not)", "{true,false}", false);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

};

}// namespace standard
}// namespace essentia


#include "streamingalgorithm.h"


namespace essentia {
namespace streaming {

class StereoTrimmer : public Algorithm {
 protected:
  Sink<StereoSample> _input;
  Source<StereoSample> _output;

  int _preferredSize;
  long long _startIndex;
  long long _endIndex;
  long long _consumed;

  static const int defaultPreferredSize = 4096;

 public:
  StereoTrimmer() : Algorithm(), _preferredSize(defaultPreferredSize) {
    declareInput(_input, _preferredSize, "signal", "the input stereo signal");
    declareOutput(_output, _preferredSize, "signal", "the trimmed stereo signal");

    _output.setBufferType(BufferUsage::forAudioStream);
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the input audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("startTime", "the start time of the slice you want to extract [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice you want to extract [s]", "[0,inf)", 1.0e6);
  }

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_STEREOTRIMMER_H
