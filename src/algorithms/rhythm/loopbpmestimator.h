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

#ifndef LOOPBPMESTIMATOR_H
#define LOOPBPMESTIMATOR_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class LoopBpmEstimator : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<Real > _bpm;
  Algorithm* _percivalBpmEstimator;
  Algorithm* _loopBpmConfidence;

 public:
  LoopBpmEstimator(){
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_bpm, "bpm", "the estimated bpm (will be 0 if unsure)");

    _percivalBpmEstimator = AlgorithmFactory::create("PercivalBpmEstimator");
    _loopBpmConfidence = AlgorithmFactory::create("LoopBpmConfidence");
  };

  ~LoopBpmEstimator() {
    delete _percivalBpmEstimator;
    delete _loopBpmConfidence;
  }

  void reset() {
    _percivalBpmEstimator->reset();
  }

  void declareParameters() {
    declareParameter("confidenceThreshold", "confidence threshold below which bpm estimate will be considered unreliable", "[0,1]", 0.95);
  }

  void compute();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class LoopBpmEstimator : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _bpm;

 public:
  LoopBpmEstimator() {
    declareAlgorithm("LoopBpmEstimator");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_bpm, TOKEN, "bpm");
  }
};

} // namespace streaming
} // namespace essentia

#endif // LOOPBPMESTIMATOR_H
