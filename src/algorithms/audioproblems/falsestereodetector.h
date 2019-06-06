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

#ifndef ESSENTIA_FALSESTEREODETECTOR_H
#define ESSENTIA_FALSESTEREODETECTOR_H

#include "algorithm.h"
#include "algorithmfactory.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {

class FalseStereoDetector : public Algorithm {

 private:
  Input<std::vector<StereoSample> > _frame;
  Output<int> _isFalseStereo;
  Output<Real> _correlation;

  Real _silenceThreshold;
  Real _correlationThreshold;

  Algorithm *_demuxer;

 public:
  FalseStereoDetector() {
    declareInput(_frame, "frame", "the input frame (must be non-empty)");
    declareOutput(_isFalseStereo, "isFalseStereo", "a flag indicating if the frame channes are simmilar");
    declareOutput(_correlation, "correlation", "correlation betweeen the input channels");
    _demuxer = AlgorithmFactory::create("StereoDemuxer");
  }

  void declareParameters() {
    declareParameter("silenceThreshold", "Silent frames will be skkiped.", "(-inf,0)", -70);
    declareParameter("correlationThreshold", "threshold to activate the isFalseStereo flag", "[-1,1]", 0.9995);
  }

  void configure();
  void compute();
  
  static const char *name;
  static const char *category;
  static const char *description;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithm.h"
#include "pool.h"

namespace essentia {
namespace streaming {

class FalseStereoDetector : public Algorithm {
 protected:
  Sink<StereoSample> _audio;
  Source<int> _isFalseStereo;
  Source<Real> _correlation;

  int  _frameSize;

  standard::Algorithm* _FalseStereoDetectorAlgo;

 public:
  FalseStereoDetector();
  ~FalseStereoDetector();

  void declareParameters() {
    declareParameter("silenceThreshold", "correation computation can be skipped if not required.", "(-inf,0)", -70);
    declareParameter("correlationThreshold", "silence threshold. Silent frames will be skkiped.", "[-1,1]", 0.9995);
    declareParameter("frameSize", "desired frame size for the analysis.", "(0,inf)", 512);
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

#endif // ESSENTIA_FALSESTEREODETECTOR_H
