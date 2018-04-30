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

#ifndef ESSENTIA_HUMDETECTOR_H
#define ESSENTIA_HUMDETECTOR_H

#include "algorithmfactory.h"
#include "network.h"
#include "pool.h"
#include "streamingalgorithmcomposite.h"

namespace essentia {
namespace streaming {

class HumDetector : public AlgorithmComposite {

 protected:
  Algorithm* _decimator;
  Algorithm* _frameCutter;
  Algorithm* _lowPass;
  Algorithm* _welch;

  SinkProxy<Real> _signal;

  Source<std::vector<Real> > _frequencies;
  Source<std::vector<Real> > _amplitudes;
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

  scheduler::Network* _network;

template< typename T >
typename std::vector<T>::iterator 
  insertSorted( std::vector<T> & vec, T const& item );


 public:
  HumDetector();
   ~HumDetector();

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_decimator));
    declareProcessStep(SingleShot(this));
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("hopSize", "the hop size with which the loudness is computed [s]", "(0,inf)", 0.1);  
    declareParameter("frameSize", "the frame size with which the loudness is computed [s]", "(0,inf)", 0.1);  
    declareParameter("timeWindow", "time to use for the hum estimation [s]", "(0,inf)",15);  
    declareParameter("Q0", "time to use for the hum estimation [s]", "(0,1)",0.1);  
    declareParameter("Q1", "time to use for the hum estimation [s]", "(0,1)",0.55);  
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
  Output<std::vector<Real> > _frequencies;
  Output<std::vector<Real> > _amplitudes;
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
       declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("hopSize", "the hop size with which the loudness is computed [s]", "(0,inf)", 0.1);  
    declareParameter("frameSize", "the frame size with which the loudness is computed [s]", "(0,inf)", 0.1);  
    declareParameter("timeWindow", "time to use for the hum estimation [s]", "(0,inf)",15);  
    declareParameter("Q0", "time to use for the hum estimation [s]", "(0,1)",0.1);  
    declareParameter("Q1", "time to use for the hum estimation [s]", "(0,1)",0.55);  
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
