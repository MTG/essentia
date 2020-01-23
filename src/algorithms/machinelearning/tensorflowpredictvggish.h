/*
 * Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_TENSORFLOWPREDICTVGGISH_H
#define ESSENTIA_TENSORFLOWPREDICTVGGISH_H


#include "streamingalgorithmcomposite.h"
#include "algorithmfactory.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class TensorflowPredictVGGish : public AlgorithmComposite {
 protected:
  Algorithm* _frameCutter;
  Algorithm* _tensorflowInputVGGish;
  Algorithm* _vectorRealToTensor;
  Algorithm* _tensorToPool;
  Algorithm* _tensorflowPredict;
  Algorithm* _poolToTensor;
  Algorithm* _tensorToVectorReal;

  SinkProxy<Real> _signal;
  SourceProxy<std::vector<Real> > _predictions;

  scheduler::Network* _network;
  bool _configured;

  void createInnerNetwork();
  void clearAlgos();

 public:
  TensorflowPredictVGGish();
  ~TensorflowPredictVGGish();

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file containing the model to use", "", Parameter::STRING);
    declareParameter("input", "the name of the input node in the TensorFlow graph", "", "model/Placeholder");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "model/Sigmoid");
    declareParameter("isTrainingName", "the name of an additional input node to indicate the model if it is in training mode or not. Leave it empty when the model does not need such input", "", "");
    declareParameter("patchHopSize", "the number of frames between the beginnings of adjacent patches. 0 to avoid overlap", "[0,inf)", 93);
    declareParameter("lastPatchMode", "what to do with the last frames. Options are to `repeat` them to fill the last patch or to `discard` them", "{discard,repeat}", "discard");
    declareParameter("accumulate", "when true it runs a single TensorFlow session at the end of the stream. Otherwise, a session is run for every new patch", "{true,false}", false);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
  }

  void configure();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "vectorinput.h"
#include "pool.h"
#include "poolstorage.h"

namespace essentia {
namespace standard {

// Standard non-streaming algorithm comes after the streaming one as it
// depends on it
class TensorflowPredictVGGish : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<std::vector<Real> > > _predictions;

  streaming::Algorithm* _tensorflowPredictVGGish;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

  void createInnerNetwork();

 public:
  TensorflowPredictVGGish();
  ~TensorflowPredictVGGish();

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file containing the model to use", "", Parameter::STRING);
    declareParameter("input", "the name of the input nodes in the Tensorflow graph", "", "model/Placeholder");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "model/Sigmoid");
    declareParameter("isTrainingName", "the name of an additional input node indicating whether the model is to be run in a training mode (for models with a training mode, leave it empty otherwise)", "", "");
    declareParameter("patchHopSize", "number of frames between the beginnings of adjacent patches. 0 to avoid overlap", "[0,inf)", 93);
    declareParameter("lastPatchMode", "what to do with the last frames. Options are to `repeat` them to fill the last patch or to discard them", "{discard,repeat}", "discard");
    declareParameter("accumulate", "when true it runs a single Tensorflow session at the end of the stream. Otherwise a session is run for every new patch", "{true,false}", false);
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_TENSORFLOWPREDICTVGGISH_H
