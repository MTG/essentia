/*
 * Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_PITCHYINPROBABILISTIC_H
#define ESSENTIA_PITCHYINPROBABILISTIC_H

#include "algorithmfactory.h"
#include "network.h"
#include "pool.h"
#include "streamingalgorithmcomposite.h"

namespace essentia {
namespace streaming {

class PitchYinProbabilistic : public AlgorithmComposite {

 protected:
  Algorithm* _frameCutter;
  Algorithm* _yinProbabilities;
  standard::Algorithm* _yinProbabilitiesHMM;

  SinkProxy<Real> _signal;
  Source<std::vector<Real> > _pitch;
  Source<std::vector<Real> > _voicedProbabilities;

  Pool _pool;

  int _frameSize;
  int _hopSize;
  Real _lowRMSThreshold;
  std::string _outputUnvoiced;
  bool _preciseTime;

  scheduler::Network* _network;


 public:
  PitchYinProbabilistic();
   ~PitchYinProbabilistic();

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
    declareProcessStep(SingleShot(this));
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size of FFT", "(0, inf)", 2048);
    declareParameter("hopSize", "the hop size with which the pitch is computed", "[1,inf)", 256); 
    declareParameter("lowRMSThreshold", "the low RMS amplitude threshold", "(0,1]", 0.1);  
    declareParameter("outputUnvoiced", "whether output unvoiced frame. zero: output non-voiced pitch as 0.; abs: output non-voiced pitch as absolute values; negative: output non-voiced pitch as negative values", "{zero,abs,negative}", "negative");
    declareParameter("preciseTime", "use non-standard precise YIN timing (slow).", "{true,false}", false);
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

class PitchYinProbabilistic : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _pitch;
  Output<std::vector<Real> > _voicedProbabilities;

  streaming::Algorithm* _PitchYinProbabilistic;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

  int _frameSize;
  int _hopSize;
  Real _lowRMSThreshold;
  std::string _outputUnvoiced;

 public:

  PitchYinProbabilistic();
  ~PitchYinProbabilistic();

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size of FFT", "(0, inf)", 2048);
    declareParameter("hopSize", "the hop size with which the pitch is computed", "[1,inf)", 256);
    declareParameter("lowRMSThreshold", "the low RMS amplitude threshold", "(0,1]", 0.1);  
    declareParameter("outputUnvoiced", "whether output unvoiced frame, zero: output non-voiced pitch as 0.; abs: output non-voiced pitch as absolute values; negative: output non-voiced pitch as negative values", "{zero,abs,negative}", "negative");
    declareParameter("preciseTime", "use non-standard precise YIN timing (slow).", "{true,false}", false);
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

#endif // ESSENTIA_LOUDNESSEBUR128_H
