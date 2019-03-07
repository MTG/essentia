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

#ifndef ESSENTIA_HUMDETECTOR_H
#define ESSENTIA_HUMDETECTOR_H

#include "algorithmfactory.h"
#include "network.h"
#include "pool.h"
#include "streamingalgorithmcomposite.h"
#include <essentia/utils/tnt/tnt2vector.h>

namespace essentia {
namespace streaming {

class HumDetector : public AlgorithmComposite {
 protected:
  Algorithm* _decimator;
  Algorithm* _frameCutter;
  Algorithm* _lowPass;
  Algorithm* _welch;
  standard::Algorithm* _Smoothing;
  standard::Algorithm* _spectralPeaks;
  standard::Algorithm* _pitchSalienceFunction;
  standard::Algorithm* _pitchSalienceFunctionPeaks;
  standard::Algorithm* _pitchContours;

  SinkProxy<Real> _signal;

  Source<TNT::Array2D<Real> > _rMatrix;
  Source<std::vector<Real> > _frequencies;
  Source<std::vector<Real> > _saliences;
  Source<std::vector<Real> > _starts;
  Source<std::vector<Real> > _ends;

  Pool _pool;
  Real _absoluteThreshold;

  uint _hopSize;
  uint _frameSize;
  uint _spectSize;
  uint _timeStamps;
  Real _sampleRate;
  Real _outSampleRate;
  Real _timeWindow;
  Real _Q0;
  Real _Q1;
  uint _Q0sample;
  uint _Q1sample;
  uint _iterations;
  uint _medianFilterSize;
  uint _numberHarmonics;
  Real _referenceTerm;
  Real _binsInOctave;
  Real _minimumFrequency;
  Real _maximumFrequency;
  Real _minDuration;
  Real _timeContinuity;
  Real _detectionThreshold;
  Real  _EPS;

  scheduler::Network* _network;

  template< typename T >
  typename std::vector<T>::iterator 
    insertSorted( std::vector<T> & vec, T const& item );

  template <typename T>
  std::vector<size_t> sort_indexes(const std::vector<T> &v);

  Real centBinToFrequency(Real cent, Real reff, Real binsInOctave);

 public:
  HumDetector();
  ~HumDetector();

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_decimator));
    declareProcessStep(SingleShot(this));
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.f);
    declareParameter("hopSize", "the hop size with which the loudness is computed [s]", "(0,inf)", 0.2f);
    declareParameter("frameSize", "the frame size with which the loudness is computed [s]", "(0,inf)", 0.4f);
    declareParameter("timeWindow", "analysis time to use for the hum estimation [s]", "(0,inf)", 9.f);
    declareParameter("minimumFrequency", "minimum frequency to consider [Hz]", "(0,inf)", 27.5f);
    declareParameter("maximumFrequency", "maximum frequency to consider [Hz]", "(0,inf)", 400.f);
    declareParameter("Q0", "low quantile", "(0,1)", 0.1f);
    declareParameter("Q1", "high quatile", "(0,1)", 0.55f);
    declareParameter("minimumDuration", "minimun duration of the humming tones [s]", "(0,inf)", 0.5f);
    declareParameter("timeContinuity", "time continuity cue (the maximum allowed gap duration for a pitch contour) [s]", "(0,inf)", 10.f);
    declareParameter("numberHarmonics", "number of considered harmonics", "(0,inf)", 1);
    declareParameter("detectionThreshold", "the detection threshold for the peaks of the r matrix", "(0,inf)", 5.f);
  };

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

class HumDetector : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<TNT::Array2D<Real> > _rMatrix;
  Output<std::vector<Real> > _frequencies;
  Output<std::vector<Real> > _saliences;
  Output<std::vector<Real> > _starts;
  Output<std::vector<Real> > _ends;

  streaming::Algorithm* _humDetector;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:
  HumDetector();
  ~HumDetector();

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.f);
    declareParameter("hopSize", "the hop size with which the loudness is computed [s]", "(0,inf)", 0.2f);
    declareParameter("frameSize", "the frame size with which the loudness is computed [s]", "(0,inf)", 0.4f);
    declareParameter("timeWindow", "analysis time to use for the hum estimation [s]", "(0,inf)", 10.f);
    declareParameter("minimumFrequency", "minimum frequency to consider [Hz]", "(0,inf)", 22.5f);
    declareParameter("maximumFrequency", "maximum frequency to consider [Hz]", "(0,inf)", 400.f);
    declareParameter("Q0", "low quantile", "(0,1)", 0.1f);
    declareParameter("Q1", "high quatile", "(0,1)", 0.55f);
    declareParameter("minimumDuration", "minimun duration of the humming tones [s]", "(0,inf)", 2.f);
    declareParameter("timeContinuity", "time continuity cue (the maximum allowed gap duration for a pitch contour) [s]", "(0,inf)", 10.f);
    declareParameter("numberHarmonics", "number of considered harmonics", "(0,inf)", 1);
    declareParameter("detectionThreshold", "the detection threshold for the peaks of the r matrix", "(0,inf)", 5.f);
  };

  void configure();
  void compute();
  void createInnerNetwork();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_HUMDETECTOR_H
