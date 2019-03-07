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

#ifndef ESSENTIA_CLICKDETECTOR_H
#define ESSENTIA_CLICKDETECTOR_H

#include "algorithm.h"
#include "algorithmfactory.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {

class ClickDetector : public Algorithm {
 private:
  Input<std::vector<Real>> _frame;
  Output<std::vector<Real>> _clickStarts;
  Output<std::vector<Real>> _clickEnds;

  int _order;
  int _frameSize;
  int _hopSize;
  Real _detectionThld;
  Real _powerEstimationThld;
  Real _silenceThld;
  Real _sampleRate;

  uint _startProc;
  uint _endProc;
  uint _idx;

  Algorithm* _LPC;
  Algorithm* _InverseFilter;
  Algorithm* _MatchedFilter;
  Algorithm* _Clipper;

  Real robustPower(std::vector<Real> x, Real k);

 public:
  ClickDetector() {
      declareInput(_frame, "frame", "the input frame (must be non-empty)");
      declareOutput(_clickStarts, "starts", "starting indexes of the clicks");
      declareOutput(_clickEnds, "ends", "ending indexes of the clicks");
  
    _LPC = AlgorithmFactory::create("LPC");
    _InverseFilter = AlgorithmFactory::create("IIR");
    _MatchedFilter = AlgorithmFactory::create("IIR");
    _Clipper = AlgorithmFactory::create("Clipper");
  }

  ~ClickDetector() {
    if (_LPC) delete _LPC;
    if (_InverseFilter) delete _InverseFilter;
    if (_MatchedFilter) delete _MatchedFilter;
    if (_Clipper) delete _Clipper;
  }

  void declareParameters() {
      declareParameter("sampleRate", "sample rate used for the analysis", "(0,inf)", 44100.);
      declareParameter("frameSize", "the expected size of the input audio signal (this is an optional parameter to optimize memory allocation)", "(0,inf)", 512);
      declareParameter("hopSize", "hop size used for the analysis. This parameter must be set correctly as it cannot be obtained from the input data", "(0,inf)", 256);
      declareParameter("order", "scalar giving the number of LPCs to use", "[1,inf)", 12);
      declareParameter("detectionThreshold", "'detectionThreshold' the threshold is based on the instant power of the noisy excitation signal plus detectionThreshold dBs", "(-inf,inf)", 30.f);
      declareParameter("powerEstimationThreshold", "the noisy excitation is clipped to 'powerEstimationThreshold' times its median.", "(0,inf)", 10);
      declareParameter("silenceThreshold", "threshold to skip silent frames", "(-inf,0)", -50);
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

class ClickDetector : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real>> _frame;
  Source<std::vector<Real>> _clickStarts;
  Source<std::vector<Real>> _clickEnds;

 public:
  ClickDetector() {
    declareAlgorithm("ClickDetector");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_clickStarts, TOKEN, "starts");
    declareOutput(_clickEnds, TOKEN, "ends");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CLICKDETECTOR_H
