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

#ifndef ESSENTIA_SUPERFLUX_EXTRACTOR_H
#define ESSENTIA_SUPERFLUX_EXTRACTOR_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "network.h"
#include "vectorinput.h"
#include "vectoroutput.h"


namespace essentia {
namespace streaming {

class SuperFluxExtractor : public AlgorithmComposite {
 
 protected:
  SinkProxy<Real> _signal;
  SourceProxy<std::vector<Real> > _onsets;

  Algorithm* _w;
  Algorithm* _spectrum;
  Algorithm* _triF;
  Algorithm* _superFluxF;
  Algorithm* _superFluxP;
  Algorithm* _frameCutter;
  Algorithm* _mfccF;
  VectorOutput<Real>* _vout;
  scheduler::Network* _network;

  bool _configured;
  void clearAlgos();

 public:
  SuperFluxExtractor();
  ~SuperFluxExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the frame size for computing low-level features", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size for computing low-level features", "(0,inf)", 256);
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.0);
    declareParameter("threshold", "threshold for peak peaking with respect to the difference between novelty_signal and average_signal (for onsets in ambient noise)", "[0,inf)", .05);
    declareParameter("ratioThreshold", "ratio threshold for peak picking with respect to novelty_signal/novelty_average rate, use 0 to disable it (for low-energy onsets)", "[0,inf)", 16.);
    declareParameter("combine", "time threshold for double onsets detections (ms)", "(0,inf)", 20.0);
  }
    
  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
  }
    
  void configure();
  void createInnerNetwork();
    
  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

class SuperFluxExtractor : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _onsets;

  bool _configured;
  streaming::Algorithm* _SuperFluxExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  streaming::VectorOutput<std::vector<Real> >* _vectorOut;
  scheduler::Network* _network;

 public:
  SuperFluxExtractor();
  ~SuperFluxExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the frame size for computing low-level features", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size for computing low-level features", "(0,inf)", 256);
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.0);
    declareParameter("threshold", "threshold for peak peaking with respect to the difference between novelty_signal and average_signal (for onsets in ambient noise)", "[0,inf)", .05);
    declareParameter("ratioThreshold", "ratio threshold for peak picking with respect to novelty_signal/novelty_average rate, use 0 to disable it (for low-energy onsets)", "[0,inf)", 16.);
    declareParameter("combine","time threshold for double onsets detections (ms)", "(0,inf)", 20.0);
  }

  void configure();
  void createInnerNetwork();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // SUPERFLUX_EXTRACTOR_H

