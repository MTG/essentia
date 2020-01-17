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

#include "tensorflowpredictmusicnn.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* TensorflowPredictMusiCNN::name = essentia::standard::TensorflowPredictMusiCNN::name;
const char* TensorflowPredictMusiCNN::category = essentia::standard::TensorflowPredictMusiCNN::category;
const char* TensorflowPredictMusiCNN::description = essentia::standard::TensorflowPredictMusiCNN::description;


TensorflowPredictMusiCNN::TensorflowPredictMusiCNN() : AlgorithmComposite(),
    _frameCutter(0), _tensorflowInputMusiCNN(0), _vectorRealToTensor(0), _tensorToPool(0),
    _tensorflowPredict(0), _poolToTensor(0), _tensorToVectorReal(0), _configured(false) {

  declareInput(_signal, 4096, "signal", "the input audio signal sampled at 16 kHz");
  declareOutput(_predictions, 0, "predictions", "the model predictions");
}


void TensorflowPredictMusiCNN::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter            = factory.create("FrameCutter");
  _tensorflowInputMusiCNN = factory.create("TensorflowInputMusiCNN");
  _vectorRealToTensor     = factory.create("VectorRealToTensor");
  _tensorToPool           = factory.create("TensorToPool");
  _tensorflowPredict      = factory.create("TensorflowPredict");
  _poolToTensor           = factory.create("PoolToTensor");
  _tensorToVectorReal     = factory.create("TensorToVectorReal");

  _tensorflowInputMusiCNN->output("bands").setBufferType(BufferUsage::forMultipleFrames);
  
  _signal                                  >> _frameCutter->input("signal");
  _frameCutter->output("frame")            >> _tensorflowInputMusiCNN->input("frame");
  _tensorflowInputMusiCNN->output("bands") >> _vectorRealToTensor->input("frame");
  _vectorRealToTensor->output("tensor")    >>  _tensorToPool->input("tensor");
  _tensorToPool->output("pool")            >>  _tensorflowPredict->input("poolIn");
  _tensorflowPredict->output("poolOut")    >>  _poolToTensor->input("pool");
  _poolToTensor->output("tensor")          >>  _tensorToVectorReal->input("tensor");

  attach(_tensorToVectorReal->output("frame"), _predictions);

  _network = new scheduler::Network(_frameCutter);
}


void TensorflowPredictMusiCNN::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


TensorflowPredictMusiCNN::~TensorflowPredictMusiCNN() {
  clearAlgos();
}


void TensorflowPredictMusiCNN::reset() {
  AlgorithmComposite::reset();
}


void TensorflowPredictMusiCNN::configure() {
  if (_configured) {
    clearAlgos();
  }

  createInnerNetwork();

  int patchHopSize = parameter("patchHopSize").toInt();
  string lastPatchMode = parameter("lastPatchMode").toString();
  bool accumulate = parameter("accumulate").toBool();

  int batchSize = accumulate ? -1 : 1;

  // Hardcoded parameters matching the training setup:
  // https://github.com/jordipons/musicnn-training/blob/master/src/config_file.py
  int frameSize = 512;
  int hopSize = 256;
  int patchSize = 187;
  int numberBands = 96;
  vector<int> inputShape({batchSize, 1, patchSize, numberBands});

  _frameCutter->configure("frameSize", frameSize, "hopSize", hopSize);

  _vectorRealToTensor->configure("shape", inputShape,
                                 "lastPatchMode", lastPatchMode,
                                 "patchHopSize", patchHopSize);

  _configured = true;

  string input = parameter("input").toString();
  string output = parameter("output").toString();
  string isTrainingName = parameter("isTrainingName").toString();

  _tensorToPool->configure("namespace", input);

  _poolToTensor->configure("namespace", output);

  Parameter graphFilenameParam = parameter("graphFilename");
  // if no file has been specified, do not do anything else
  if (!graphFilenameParam.isConfigured()) return;

  string graphFilename = parameter("graphFilename").toString();

  _tensorflowPredict->configure("graphFilename", graphFilename,
                                "inputs", vector<string>({input}),
                                "outputs", vector<string>({output}),
                                "isTrainingName", isTrainingName);
}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* TensorflowPredictMusiCNN::name = "TensorflowPredictMusiCNN";
const char* TensorflowPredictMusiCNN::category = "Machine Learning";
const char* TensorflowPredictMusiCNN::description = DOC(
  "This algorithm makes predictions using MusiCNN-based models [1, 2].\n"
  "Internally, it uses TensorflowInputMusiCNN for the input feature extraction (mel bands). "
  "It feeds the model with patches of 187 mel bands frames and jumps a constant amount of frames determined by patchHopSize.\n"
  "With the `accumulate` parameter the patches are stored to run a single TensorFlow session at the end of the stream. "
  "This allows to take advantage of parallelization when GPUs are available, but at the same time it can be memory exhausting for long files.\n"
  "The recommended pipeline is as follows:\n"
  "  MonoLoader(sampleRate=16000) >> TensorflowPredictMusiCNN"
  "\n"
  "Note: This algorithm does not make any check on the input model so it is the user's responsibility to make sure it is a valid one."
  "\n"
  "References:\n"
  "  [1] Pons, J., & Serra, X. (2019). musicnn: Pre-trained convolutional neural networks for music audio tagging. arXiv preprint arXiv:1909.06654.\n"
  "  [2] Supported models at https://essentia.upf.edu/models/");


TensorflowPredictMusiCNN::TensorflowPredictMusiCNN() {
    declareInput(_signal, "signal", "the input audio signal sampled at 16 kHz");
    declareOutput(_predictions, "predictions", "the predictions");

    createInnerNetwork();
  }


TensorflowPredictMusiCNN::~TensorflowPredictMusiCNN() {
  delete _network;
}


void TensorflowPredictMusiCNN::createInnerNetwork() {
  _tensorflowPredictMusiCNN = streaming::AlgorithmFactory::create("TensorflowPredictMusiCNN");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >> _tensorflowPredictMusiCNN->input("signal");
  _tensorflowPredictMusiCNN->output("predictions") >>  PC(_pool, "predictions");

  _network = new scheduler::Network(_vectorInput);
}


void TensorflowPredictMusiCNN::configure() {
  // if no file has been specified, do not do anything
  if (!parameter("graphFilename").isConfigured()) return;
  _tensorflowPredictMusiCNN->configure(INHERIT("graphFilename"),
                                       INHERIT("input"),
                                       INHERIT("output"),
                                       INHERIT("isTrainingName"),
                                       INHERIT("patchHopSize"),
                                       INHERIT("accumulate"),
                                       INHERIT("lastPatchMode"));
}


void TensorflowPredictMusiCNN::compute() {
  const vector<Real>& signal = _signal.get();
  vector<vector<Real> >& predictions = _predictions.get();

  if (!signal.size()) {
    throw EssentiaException("TensorflowPredictMusiCNN: empty input signal");
  }

  _vectorInput->setVector(&signal);

  _network->run();

  try {
    predictions = _pool.value<vector<vector<Real> > >("predictions");
  }
  catch (EssentiaException&) {
    predictions.clear();
  }

  reset();
}


void TensorflowPredictMusiCNN::reset() {
  _network->reset();
  _pool.remove("predictions");
}

} // namespace standard
} // namespace essentia
