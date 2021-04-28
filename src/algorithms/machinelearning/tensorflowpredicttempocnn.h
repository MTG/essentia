/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_TENSORFLOWPREDICTTEMPOCNN_H
#define ESSENTIA_TENSORFLOWPREDICTTEMPOCNN_H


#include "streamingalgorithmcomposite.h"
#include "algorithmfactory.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class TensorflowPredictTempoCNN : public AlgorithmComposite {
 protected:
  Algorithm* _frameCutter;
  Algorithm* _tensorflowInputTempoCNN;
  Algorithm* _vectorRealToTensor;
  Algorithm* _tensorNormalize;
  Algorithm* _tensorTranspose;
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
  TensorflowPredictTempoCNN();
  ~TensorflowPredictTempoCNN();

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file containing the model to use", "", Parameter::STRING);
    declareParameter("input", "the name of the input node in the TensorFlow graph", "", "input");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "output");
    declareParameter("patchHopSize", "the number of frames between the beginnings of adjacent patches. 0 to avoid overlap", "[0,inf)", 128);
    declareParameter("lastPatchMode", "what to do with the last frames: `repeat` them to fill the last patch or `discard` them", "{discard,repeat}", "discard");
    declareParameter("batchSize", "number of patches to process in parallel. Use -1 to accumulate all the patches and run a single TensorFlow session at the end of the stream.", "[-1,inf)", 1);
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
class TensorflowPredictTempoCNN : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<std::vector<Real> > > _predictions;

  streaming::Algorithm* _tensorflowPredictTempoCNN;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

  void createInnerNetwork();

 public:
  TensorflowPredictTempoCNN();
  ~TensorflowPredictTempoCNN();

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file containing the model to use", "", Parameter::STRING);
    declareParameter("input", "the name of the input nodes in the Tensorflow graph", "", "input");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "output");
    declareParameter("patchHopSize", "number of frames between the beginnings of adjacent patches. 0 to avoid overlap", "[0,inf)", 128);
    declareParameter("lastPatchMode", "what to do with the last frames: `repeat` them to fill the last patch or `discard` them", "{discard,repeat}", "discard");
    declareParameter("batchSize", "number of patches to process in parallel. Use -1 to accumulate all the patches and run a single TensorFlow session at the end of the stream.", "[-1,inf)", 1);
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

#endif // ESSENTIA_TENSORFLOWPREDICTTEMPOCNN_H
