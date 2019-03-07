/*
 * Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_TRUEPEAKDETECTOR_H
#define ESSENTIA_TRUEPEAKDETECTOR_H

#include "algorithmfactory.h"
#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace standard {

class TruePeakDetector : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _output;
  Output<std::vector<Real> > _peakLocations;

  Algorithm* _resampler;
  Algorithm* _emphasiser;
  Algorithm* _dcBlocker;

  Real _inputSampleRate;
  Real _outputSampleRate;
  Real _oversamplingFactor;
  int _quality;
  bool _blockDC;
  bool _emphasise;
  Real _threshold;
  uint _version;

 public:
  TruePeakDetector() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_peakLocations, "peakLocations", "the peak locations in the ouput signal");
    declareOutput(_output, "output", "the processed signal");

    _resampler = AlgorithmFactory::create("Resample");
    _emphasiser = AlgorithmFactory::create("IIR");
    _dcBlocker = AlgorithmFactory::create("DCRemoval");
  }

  ~TruePeakDetector() {
    if (_resampler) delete _resampler;
    if (_emphasiser) delete _emphasiser;
    if (_dcBlocker) delete _dcBlocker;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("oversamplingFactor", "times the signal is oversapled", "[1,inf)", 4);
    declareParameter("quality", "type of interpolation applied (see libresmple)", "[0,4]", 1);
    declareParameter("blockDC", "flag to activate the optional DC blocker", "{true,false}", false);
    declareParameter("emphasise", "flag to activate the optional emphasis filter", "{true,false}", false);
    declareParameter("threshold", "threshold to detect peaks [dB]", "(-inf,inf)", -0.0002);
    declareParameter("version", "algorithm version", "{2,4}",4);
  }

  void reset() {
    _resampler->reset();
    _emphasiser->reset();
    _dcBlocker->reset();
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard


namespace streaming {

class TruePeakDetector : public StreamingAlgorithmWrapper {
 protected:
  Sink<std::vector<Real> > _signal;
  Source<std::vector<Real> > _output;
  Source<std::vector<Real> > _peakLocations;

 public:
  TruePeakDetector() {
    declareAlgorithm("TruePeakDetector");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_output, TOKEN, "output");
    declareOutput(_peakLocations, TOKEN, "peakLocations");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TRUEPEAKDETECTOR_H
