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

#ifndef ESSENTIA_RESAMPLE_H
#define ESSENTIA_RESAMPLE_H

#include <samplerate.h>
#include "algorithm.h"

namespace essentia {
namespace standard {

class Resample : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _resampled;

 public:
  Resample() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_resampled, "signal", "the resampled signal");
  }

  void declareParameters() {
    declareParameter("inputSampleRate", "the sampling rate of the input signal [Hz]", "(0,inf)", 44100.);
    declareParameter("outputSampleRate", "the sampling rate of the output signal [Hz]", "(0,inf)", 44100.);
    declareParameter("quality", "the quality of the conversion, 0 for best quality", "[0,4]", 1);
  }

  void configure();

  void compute();

  static const char* name;
  static const char* description;

 protected:
  double _factor;
  int _quality;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class Resample : public Algorithm {

 protected:
  Sink<Real> _signal;
  Source<Real> _resampled;
  int _preferredSize;

  SRC_STATE* _state;
  SRC_DATA _data;
  int _errorCode;
  float _delay;

 public:
  Resample() : _state(0) {
    _preferredSize = 4096; // arbitrary
    declareInput(_signal, _preferredSize, "signal", "the input signal");
    declareOutput(_resampled, _preferredSize, "signal", "the resampled signal");

    // useless as we do it anyway in the configure() method
    //_resampled.setBufferType(BufferUsage::forAudioStream);
  }

  ~Resample();

  void declareParameters() {
    declareParameter("inputSampleRate", "the sampling rate of the input signal [Hz]", "(0,inf)", 44100.);
    declareParameter("outputSampleRate", "the sampling rate of the output signal [Hz]", "(0,inf)", 44100.);
    declareParameter("quality", "the quality of the conversion, 0 for best quality", "[0,4]", 1);
  }

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_RESAMPLE_H
