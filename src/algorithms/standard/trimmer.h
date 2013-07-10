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

#ifndef ESSENTIA_TRIMMER_H
#define ESSENTIA_TRIMMER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Trimmer : public Algorithm {

 private:
  Input<std::vector<Real> > _input;
  Output<std::vector<Real> > _output;

  long long _startIndex;
  long long _endIndex;

 public:
  Trimmer() {
    declareInput(_input, "signal", "the input signal");
    declareOutput(_output, "signal", "the trimmed signal");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the input audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("startTime", "the start time of the slice you want to extract [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice you want to extract [s]", "[0,inf)", 1.0e6);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

}// namespace standard
}// namespace essentia


#include "streamingalgorithm.h"


namespace essentia {
namespace streaming {

class Trimmer : public Algorithm {
 protected:
  Sink<Real> _input;
  Source<Real> _output;

  int _preferredSize;
  long long _startIndex;
  long long _endIndex;
  long long _consumed;

  static const int defaultPreferredSize = 4096;

 public:
  Trimmer() : Algorithm(), _preferredSize(defaultPreferredSize) {
    declareInput(_input, _preferredSize, "signal", "the input signal");
    declareOutput(_output, _preferredSize, "signal", "the trimmed signal");

    _output.setBufferType(BufferUsage::forAudioStream);
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the input audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("startTime", "the start time of the slice you want to extract [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice you want to extract [s]", "[0,inf)", 1.0e6);
  }

  void configure();
  AlgorithmStatus process();

  void reset() {
    Algorithm::reset();
    _consumed = 0;
  }

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_TRIMMER_H
