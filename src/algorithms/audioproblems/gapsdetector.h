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

#ifndef ESSENTIA_GAPSDETECTOR_H
#define ESSENTIA_GAPSDETECTOR_H

#include "algorithm.h"
#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class GapsDetector : public Algorithm {
 private:
  Input<std::vector<Real>> _frame;
  Output<std::vector<Real>> _gapsStarts;
  Output<std::vector<Real>> _gapsEnds;

  struct gap {
    uint remaining;
    Real start, end;
    bool active, finished;
    std::vector<Real> rBuffer;
  };

  uint _frameSize, _hopSize;
  uint _prepowerSamples, _postpowerSamples;
  uint _updateSize;
  long _frameCount;
  Real _sampleRate;
  Real _silenceThreshold;
  Real _prepowerThreshold;
  Real _prepowerTime, _postpowerTime, _minimumTime, _maximumTime;
  std::vector<Real> _lBuffer;
  std::vector<gap> _gaps;

  Algorithm *_medianFilter;
  Algorithm *_envelope;

 public:
  GapsDetector() {
    declareInput(_frame, "frame", "the input frame (must be non-empty)");
    declareOutput(_gapsStarts, "starts", "the start indexes of the detected gaps (if any) in seconds");
    declareOutput(_gapsEnds, "ends", "the end indexes of the detected gaps (if any) in seconds");

    _medianFilter = AlgorithmFactory::create("MedianFilter");
    _envelope = AlgorithmFactory::create("Envelope");
  }

  ~GapsDetector() {
    if (_medianFilter) delete _medianFilter;
    if (_envelope) delete _envelope;
  }

  void declareParameters() {
    declareParameter("sampleRate", "sample rate used for the analysis", "(0,inf)", 44100.f);
    declareParameter("frameSize", "frame size used for the analysis. Should match the input frame size. Otherwise, an exception will be thrown", "[0,inf)", 2048);
    declareParameter("hopSize", "hop size used for the analysis", "[0,inf)", 1024);
    declareParameter("silenceThreshold", "silence threshold [dB]", "(-inf,inf)", -50.f);
    declareParameter("prepowerThreshold", "prepower threshold [dB]. ", "(-inf,inf)", -30.f);
    declareParameter("prepowerTime", "time for the prepower calculation [ms]", "(0,inf)", 40.f);
    declareParameter("postpowerTime", "time for the postpower calculation [ms]", "(0,inf)", 40.f);
    declareParameter("minimumTime", "time of the minimum gap duration [ms]", "(0,inf)", 10.f);
    declareParameter("maximumTime", "time of the maximum gap duration [ms]", "(0,inf)", 3500.f);
    declareParameter("kernelSize", "scalar giving the size of the median filter window. Must be odd", "[1,inf)", 11);
    declareParameter("attackTime", "the attack time of the first order lowpass in the attack phase [ms]", "[0,inf)", 0.05);
    declareParameter("releaseTime", "the release time of the first order lowpass in the release phase [ms]", "[0,inf)", 0.05);
  }

    void configure();
    void compute();
    void reset();

    static const char *name;
    static const char *category;
    static const char *description;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class GapsDetector : public StreamingAlgorithmWrapper {
 protected:
  Sink<std::vector<Real>> _frame;
  Source<std::vector<Real>> _gapsStarts;
  Source<std::vector<Real>> _gapsEnds;

 public:
  GapsDetector() {
    declareAlgorithm("GapsDetector");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_gapsStarts, TOKEN, "starts");
    declareOutput(_gapsEnds, TOKEN, "ends");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_GAPSDETECTOR_H
