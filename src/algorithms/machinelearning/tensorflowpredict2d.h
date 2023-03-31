/*
 * Copyright (C) 2006-2022  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_TENSORFLOWPREDICT2D_H
#define ESSENTIA_TENSORFLOWPREDICT2D_H


#include "streamingalgorithmcomposite.h"
#include "algorithmfactory.h"
#include "algorithm.h"
#include "network.h"
#include "tnt/tnt.h"

namespace essentia {
namespace streaming {

class TensorflowPredict2D : public AlgorithmComposite {
 protected:
  Algorithm* _vectorRealToTensor;
  Algorithm* _tensorToPool;
  Algorithm* _tensorflowPredict;
  Algorithm* _poolToTensor;
  Algorithm* _tensorToVectorReal;

  SinkProxy<std::vector<Real> > _features;
  SourceProxy<std::vector<Real> > _predictions;

  scheduler::Network* _network;
  bool _configured;

  void createInnerNetwork();
  void clearAlgos();

 public:
  TensorflowPredict2D();
  ~TensorflowPredict2D();

  void declareParameters() {
    declareParameter("graphFilename", "name of the file from which to load the TensorFlow graph", "", "");
    declareParameter("savedModel", "name of the TensorFlow SavedModel. Overrides the parameter `graphFilename`", "", "");
    declareParameter("input", "name of the input node in the TensorFlow graph", "", "model/Placeholder");
    declareParameter("output", "name of the node from which to retrieve the output tensors", "", "model/Sigmoid");
    declareParameter("isTrainingName", "name of an additional input node to indicate the model if it is in training mode or not. Leave it empty when the model does not need such input", "", "");
    declareParameter("patchHopSize", "number of timestamps between the beginning of adjacent patches. 0 to avoid overlap", "[0,inf)", 1);
    declareParameter("lastPatchMode", "what to do with the last timestamps: `repeat` them to fill the last patch or `discard` them", "{discard,repeat}", "discard");
    declareParameter("accumulate", "(deprecated, use `batchSize`) when true it runs a single TensorFlow session at the end of the stream. Otherwise, a session is run for every new patch", "{true,false}", false);
    declareParameter("batchSize", "batch size for prediction. This allows parallelization when GPUs are available. Set it to -1 or 0 to accumulate all the patches and run a single TensorFlow session at the end of the stream", "[-1,inf)", 64);
    declareParameter("patchSize", "number of timestamps required for each inference. This parameter should match the model's expected input shape.", "[0,inf)", 1);
    declareParameter("dimensions", "number of dimensions on the input features. This parameter should match the model's expected input shape", "[0,inf)", 200);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_vectorRealToTensor));
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
class TensorflowPredict2D : public Algorithm {
 protected:
  Input<TNT::Array2D<Real> > _features;
  Output<std::vector<std::vector<Real> > > _predictions;

  streaming::Algorithm* _tensorflowPredict2D;
  streaming::VectorInput<std::vector<Real> >* _vectorVectorInput;
  scheduler::Network* _network;
  Pool _pool;

  int _dimensions;

  void createInnerNetwork();

 public:
  TensorflowPredict2D();
  ~TensorflowPredict2D();

  void declareParameters() {
    declareParameter("graphFilename", "name of the file from which to load the TensorFlow graph", "", "");
    declareParameter("savedModel", "name of the TensorFlow SavedModel. Overrides the parameter `graphFilename`", "", "");
    declareParameter("input", "name of the input node in the TensorFlow graph", "", "model/Placeholder");
    declareParameter("output", "name of the node from which to retrieve the output tensors", "", "model/Sigmoid");
    declareParameter("isTrainingName", "name of an additional input node to indicate the model if it is in training mode or not. Leave it empty when the model does not need such input", "", "");
    declareParameter("patchHopSize", "number of timestamps between the beginning of adjacent patches. 0 to avoid overlap", "[0,inf)", 1);
    declareParameter("lastPatchMode", "what to do with the last timestamps: `repeat` them to fill the last patch or `discard` them", "{discard,repeat}", "discard");
    declareParameter("accumulate", "(deprecated, use `batchSize`) when true it runs a single TensorFlow session at the end of the stream. Otherwise, a session is run for every new patch", "{true,false}", false);
    declareParameter("batchSize", "batch size for prediction. This allows parallelization when GPUs are available. Set it to -1 or 0 to accumulate all the patches and run a single TensorFlow session at the end of the stream", "[-1,inf)", 64);
    declareParameter("patchSize", "number of timestamps required for each inference. This parameter should match the model's expected input shape.", "[0,inf)", 1);
    declareParameter("dimensions", "number of dimensions on the input features. This parameter is overridden by the shape of the input data", "[0,inf)", 200);
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

#endif // ESSENTIA_TENSORFLOWPREDICT2D_H
