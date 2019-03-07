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

#ifndef ESSENTIA_DISCONTINUITYDETECTOR_H
#define ESSENTIA_DISCONTINUITYDETECTOR_H

#include "algorithm.h"
#include "algorithmfactory.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {

class DiscontinuityDetector : public Algorithm {

 private:
  Input<std::vector<Real>> _frame;
  Output<std::vector<Real>> _discontinuityLocations;
  Output<std::vector<Real>> _discontinuityAmplitues;

  int _order;
  int _hopSize;
  int _kernelSize;
  float _detectionThld;
  float _energyThld;
  int _subFrameSize;
  int _frameSize;
  float _silenceThld;

  Algorithm* _medianFilter;
  Algorithm* _LPC;
  Algorithm* _windowing;

 public:
  DiscontinuityDetector() {
    declareInput(_frame, "frame", "the input frame (must be non-empty)");
    declareOutput(_discontinuityLocations, "discontinuityLocations", "the index of the detected discontinuities (if any)");
    declareOutput(_discontinuityAmplitues, "discontinuityAmplitudes", "the peak values of the prediction error for the discontinuities (if any)");

    _medianFilter = AlgorithmFactory::create("MedianFilter");
    _LPC = AlgorithmFactory::create("LPC");
    _windowing = AlgorithmFactory::create("Windowing");
  }

  ~DiscontinuityDetector() {
    if (_medianFilter) delete _medianFilter;
    if (_LPC) delete _LPC;
    if (_windowing) delete _windowing;
  }

  void declareParameters() {
    declareParameter("order", "scalar giving the number of LPCs to use", "[1,inf)", 3);
    declareParameter("frameSize", "the expected size of the input audio signal (this is an optional parameter to optimize memory allocation)", "(0,inf)", 512);
    declareParameter("hopSize", "hop size used for the analysis. This parameter must be set correctly as it cannot be obtained from the input data", "[0,inf)", 256);
    declareParameter("kernelSize", "scalar giving the size of the median filter window. Must be odd", "[1,inf)", 7);
    declareParameter("detectionThreshold", "'detectionThreshold' times the standard deviation plus the median of the frame is used as detection threshold", "[1,inf)", 8.f);
    declareParameter("energyThreshold", "threshold in dB to detect silent subframes", "(-inf,inf)", -60.f);
    declareParameter("subFrameSize", "size of the window used to compute silent subframes", "[1,inf)", 32);
    declareParameter("silenceThreshold", "threshold to skip silent frames", "(-inf,0)", -50);
  }

  void configure();
  void compute();

  static const char *name;
  static const char *category;
  static const char *description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class DiscontinuityDetector : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real>> _frame;
  Source<std::vector<Real>> _discontinuityLocations;
  Source<std::vector<Real>> _discontinuityAmplitues;

 public:
  DiscontinuityDetector() {
    declareAlgorithm("DiscontinuityDetector");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_discontinuityLocations, TOKEN, "discontinuityLocations");
    declareOutput(_discontinuityAmplitues, TOKEN, "discontinuityAmplitudes");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DISCONTINUITYDETECTOR_H
