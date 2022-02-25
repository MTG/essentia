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

#ifndef ESSENTIA_TENSORFLOWPREDICTEFFNETDISCOGS_H
#define ESSENTIA_TENSORFLOWPREDICTEFFNETDISCOGS_H


#include "streamingalgorithmcomposite.h"
#include "algorithmfactory.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class TensorflowPredictEffnetDiscogs : public AlgorithmComposite {
 protected:
  Algorithm* _frameCutter;
  Algorithm* _tensorflowInputMusiCNN;
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

  // Hardcoded parameters matching the training setup. We used MusiCNN style mel-spectrograms:
  // https://github.com/jordipons/musicnn-training/blob/master/src/config_file.py
  const int _frameSize = 512;
  const int _hopSize = 256;
  const int _numberBands = 96;

 public:
  TensorflowPredictEffnetDiscogs();
  ~TensorflowPredictEffnetDiscogs();

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file from which to load the TensorFlow graph", "", "");
    declareParameter("savedModel", "the name of the TensorFlow SavedModel. Overrides parameter `graphFilename`", "", "");
    declareParameter("input", "the name of the input node in the TensorFlow graph", "", "serving_default_melspectrogram");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "PartitionedCall");
    declareParameter("patchHopSize", "the number of frames between the beginnings of adjacent patches. 0 to avoid overlap. The default value is 62 frames which corresponds to a prediction rate of 1.008 Hz", "[0,inf)", 62);
    declareParameter("lastPatchMode", "what to do with the last frames: `repeat` them to fill the last patch or `discard` them", "{discard,repeat}", "discard");
    declareParameter("batchSize", "the batch size for prediction. This allows parallelization when GPUs are available. Set it to -1 or 0 to accumulate all the patches and run a single TensorFlow session at the end of the stream", "[-1,inf)", 64);
    declareParameter("patchSize", "number of frames required for each inference. This parameter should match the model's expected input shape.", "[0,inf)", 128);
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
class TensorflowPredictEffnetDiscogs : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<std::vector<Real> > > _predictions;

  streaming::Algorithm* _tensorflowPredictEffnetDiscogs;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

  int _batchSize;
  int _patchSize;
  int _patchHopSize;
  std::string _lastPatchMode;
  std::string _lastBatchMode;

  // Hardcoded parameters matching the training setup. We used MusiCNN style mel-spectrograms:
  // https://github.com/jordipons/musicnn-training/blob/master/src/config_file.py
  const int _frameSize = 512;
  const int _hopSize = 256;
  const int _sampleRate = 16000;

  void createInnerNetwork();
  int padSignal(const std::vector<Real> &signal, std::vector<Real> &paddedSignal);

 public:
  TensorflowPredictEffnetDiscogs();
  ~TensorflowPredictEffnetDiscogs();

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file from which to load the TensorFlow graph", "", "");
    declareParameter("savedModel", "the name of the TensorFlow SavedModel. Overrides parameter `graphFilename`", "", "");
    declareParameter("input", "the name of the input node in the TensorFlow graph", "", "serving_default_melspectrogram");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "PartitionedCall");
    declareParameter("patchHopSize", "the number of frames between the beginnings of adjacent patches. 0 to avoid overlap. The default value is 62 frames which corresponds to a prediction rate of 1.008 Hz", "[0,inf)", 62);
    declareParameter("lastPatchMode", "what to do with the last frames: `repeat` them to fill the last patch or `discard` them", "{discard,repeat}", "discard");
    declareParameter("batchSize", "the batch size for prediction. This allows parallelization when GPUs are available. Set it to -1 or 0 to accumulate all the patches and run a single TensorFlow session at the end of the stream", "[-1,inf)", 64);
    declareParameter("patchSize", "number of frames required for each inference. This parameter should match the model's expected input shape.", "[0,inf)", 128);
    declareParameter("lastBatchMode", "some EffnetDiscogs models operate on a fixed batch size. The options are to `discard` the last patches or to pad with `zeros` to make a final batch. Additionally `same` zero-pads the input but returns only the predictions corresponding to patches with signal", "{discard,zeros,same}", "same");
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

#endif // ESSENTIA_TENSORFLOWPREDICTEFFNETDISCOGS_H
