/*
 * Copyright (C) 2006-2023  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_TENSORFLOWPREDICTMAEST_H
#define ESSENTIA_TENSORFLOWPREDICTMAEST_H


#include "streamingalgorithmcomposite.h"
#include "algorithmfactory.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class TensorflowPredictMAEST : public AlgorithmComposite {
 protected:
  Algorithm* _frameCutter;
  Algorithm* _tensorflowInputMusiCNN;
  Algorithm* _shift;
  Algorithm* _scale;
  Algorithm* _vectorRealToTensor;
  Algorithm* _tensorToPool;
  Algorithm* _tensorflowPredict;
  Algorithm* _poolToTensor;

  SinkProxy<Real> _signal;
  SourceProxy<Tensor<Real> > _predictions;

  scheduler::Network* _network;
  bool _configured;

  void createInnerNetwork();
  void clearAlgos();

  // Hardcoded parameters similar to our previous models (EffnetDiscogs/MusiCNN).
  int _frameSize = 512;
  int _hopSize = 256;
  int _numberBands = 96;
  double _mean = 2.06755686098554;
  double _std = 1.268292820667291;

 public:
  TensorflowPredictMAEST();
  ~TensorflowPredictMAEST();

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file from which to load the TensorFlow graph", "", "");
    declareParameter("savedModel", "the name of the TensorFlow SavedModel. Overrides parameter `graphFilename`", "", "");
    declareParameter("input", "the name of the input node in the TensorFlow graph", "", "melspectrogram");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "Identity");
    declareParameter("isTrainingName", "the name of an additional input node to indicate the model if it is in training mode or not. Leave it empty when the model does not need such input", "", "");
    declareParameter("patchHopSize", "the number of frames between the beginnings of adjacent patches. 0 to avoid overlap", "[0,inf)", 1875);
    declareParameter("lastPatchMode", "what to do with the last frames: `repeat` them to fill the last patch or `discard` them", "{discard,repeat}", "discard");
    declareParameter("batchSize", "the batch size for prediction. This allows parallelization when GPUs are available. Set it to -1 or 0 to accumulate all the patches and run a single TensorFlow session at the end of the stream", "[-1,inf)", 1);
    declareParameter("patchSize", "number of frames required for each inference. This parameter should match the model's expected input shape.", "[0,inf)", 1876);
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
class TensorflowPredictMAEST : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<Tensor<Real> > _predictions;

  streaming::Algorithm* _tensorflowPredictMAEST;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

  void createInnerNetwork();

 public:
  TensorflowPredictMAEST();
  ~TensorflowPredictMAEST();

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file from which to load the TensorFlow graph", "", "");
    declareParameter("savedModel", "the name of the TensorFlow SavedModel. Overrides parameter `graphFilename`", "", "");
    declareParameter("input", "the name of the input nodes in the Tensorflow graph", "", "melspectrogram");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "Identity");
    declareParameter("isTrainingName", "the name of an additional input node indicating whether the model is to be run in a training mode (for models with a training mode, leave it empty otherwise)", "", "");
    declareParameter("patchHopSize", "number of frames between the beginnings of adjacent patches. 0 to avoid overlap", "[0,inf)", 1875);
    declareParameter("lastPatchMode", "what to do with the last frames: `repeat` them to fill the last patch or `discard` them", "{discard,repeat}", "discard");
    declareParameter("batchSize", "the batch size for prediction. This allows parallelization when GPUs are available. Set it to -1 or 0 to accumulate all the patches and run a single TensorFlow session at the end of the stream", "[-1,inf)", 1);
    declareParameter("patchSize", "number of frames required for each inference. This parameter should match the model's expected input shape.", "[0,inf)", 1876);
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

#endif // ESSENTIA_TENSORFLOWPREDICTMAEST_H
