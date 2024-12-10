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

#ifndef ESSENTIA_STEREORESAMPLE_H
#define ESSENTIA_STEREORESAMPLE_H

#include <samplerate.h>
#include "algorithm.h"

namespace essentia {
namespace standard {

class StereoResample : public Algorithm {

 protected:
  Input<std::vector<StereoSample> > _signal;
  Output<std::vector<StereoSample> > _StereoResampled;

 public:
  StereoResample() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_StereoResampled, "signal", "the StereoResampled signal");
  }

  void declareParameters() {
    declareParameter("inputSampleRate", "the sampling rate of the input signal [Hz]", "(0,inf)", 44100.);
    declareParameter("outputSampleRate", "the sampling rate of the output signal [Hz]", "(0,inf)", 44100.);
    declareParameter("quality", "the quality of the conversion, 0 for best quality, 4 for fast linear approximation", "[0,4]", 1);
  }

  void configure();

  void compute();

  static const char* name;
  static const char* category;
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

class StereoResample : public Algorithm {

 protected:
  Sink<StereoSample> _signal;
  Source<StereoSample> _resampled;
  int _preferredSize;

  SRC_STATE* _state;
  SRC_DATA _data;
  int _errorCode;
  float _delay;

 public:
  StereoResample() : _state(0) {
    _preferredSize = 4096; // arbitrary
    declareInput(_signal, _preferredSize, "signal", "the input stereo signal");
    declareOutput(_resampled, _preferredSize, "signal", "the stereo resampled signal");

    // useless as we do it anyway in the configure() method
    //_StereoResampled.setBufferType(BufferUsage::forAudioStream);
  }

  ~StereoResample();

  void declareParameters() {
    declareParameter("inputSampleRate", "the sampling rate of the input stereo signal [Hz]", "(0,inf)", 44100.);
    declareParameter("outputSampleRate", "the sampling rate of the output stereo signal [Hz]", "(0,inf)", 44100.);
    declareParameter("quality", "the quality of the conversion, 0 for best quality, 4 for fast linear approximation", "[0,4]", 1);
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

#endif // ESSENTIA_STEREORESAMPLE_H
