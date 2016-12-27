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

#ifndef PERCIVALBPMESTIMATOR_H
#define PERCIVALBPMESTIMATOR_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class PercivalBpmEstimator : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _signal;
  Source<Real> _bpm;

  Pool _pool;
  int _sampleRate;
  int _frameSize;
  int _hopSize;
  int _frameSizeOSS;
  int _hopSizeOSS;
  int _minBPM;
  int _maxBPM;
  Real _srOSS;

  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _spectrum;
  Algorithm* _normSpectrum;
  Algorithm* _scaleSpectrum;
  Algorithm* _shiftSpectrum;
  Algorithm* _logSpectrum;
  Algorithm* _flux;
  Algorithm* _lowPass;
  Algorithm* _frameCutterOSS;
  Algorithm* _autoCorrelation;
  Algorithm* _enhanceHarmonics;
  Algorithm* _peakDetection;
  Algorithm* _evaluatePulseTrains;

  scheduler::Network* _network;
  bool _configured;
  void createInnerNetwork();
  void clearAlgos();
  Real energyInRange(const std::vector<Real>& array,
                     const Real low,
                     const Real high,
                     const Real scale);

 public:
  PercivalBpmEstimator();

  ~PercivalBpmEstimator();

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100);
    declareParameter("frameSize", "frame size for the analysis of the input signal", "(0,inf)", 1024);
    declareParameter("hopSize", "hop size for the analysis of the input signal", "(0,inf)", 128);
    declareParameter("frameSizeOSS", "frame size for the analysis of the Onset Strength Signal", "(0,inf)", 2048);
    declareParameter("hopSizeOSS", "hop size for the analysis of the Onset Strength Signal", "(0,inf)", 128);
    declareParameter("minBPM", "minimum BPM to detect", "(0,inf)", 50);
    declareParameter("maxBPM", "maximum BPM to detect", "(0,inf)", 210);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
    declareProcessStep(SingleShot(this));
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

#include "vectorinput.h"

namespace essentia {
namespace standard {

class PercivalBpmEstimator : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<Real > _bpm;

  streaming::Algorithm* _percivalBpmEstimator;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  PercivalBpmEstimator();
  ~PercivalBpmEstimator();

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100);
    declareParameter("frameSize", "frame size for the analysis of the input signal", "(0,inf)", 1024);
    declareParameter("hopSize", "hop size for the analysis of the input signal", "(0,inf)", 128);
    declareParameter("frameSizeOSS", "frame size for the analysis of the Onset Strength Signal", "(0,inf)", 2048);
    declareParameter("hopSizeOSS", "hop size for the analysis of the Onset Strength Signal", "(0,inf)", 128);
    declareParameter("minBPM", "minimum BPM to detect", "(0,inf)", 50);
    declareParameter("maxBPM", "maximum BPM to detect", "(0,inf)", 210);
  }

  void configure();
  void compute();
  void reset();
  void createInnerNetwork();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia


#endif // PERCIVALBPMESTIMATOR_H
