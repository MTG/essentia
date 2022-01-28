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

#include "tensorflowpredictcrepe.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* TensorflowPredictCREPE::name = essentia::standard::TensorflowPredictCREPE::name;
const char* TensorflowPredictCREPE::category = essentia::standard::TensorflowPredictCREPE::category;
const char* TensorflowPredictCREPE::description = essentia::standard::TensorflowPredictCREPE::description;


TensorflowPredictCREPE::TensorflowPredictCREPE() : AlgorithmComposite(),
    _frameCutter(0), _vectorRealToTensor(0), _tensorNormalize(0), _tensorToPool(0),
    _tensorflowPredict(0), _poolToTensor(0), _tensorToVectorReal(0), _configured(false) {

  declareInput(_signal, 4096, "signal", "the input audio signal sampled at 16 kHz");
  declareOutput(_predictions, 0, "predictions", "the output values from the model node named after `output`");
}


void TensorflowPredictCREPE::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter            = factory.create("FrameCutter");
  _vectorRealToTensor     = factory.create("VectorRealToTensor");
  _tensorNormalize        = factory.create("TensorNormalize");
  _tensorToPool           = factory.create("TensorToPool");
  _tensorflowPredict      = factory.create("TensorflowPredict");
  _poolToTensor           = factory.create("PoolToTensor");
  _tensorToVectorReal     = factory.create("TensorToVectorReal");

  _signal                                  >> _frameCutter->input("signal");
  _frameCutter->output("frame")            >> _vectorRealToTensor->input("frame");
  _vectorRealToTensor->output("tensor")    >> _tensorNormalize->input("tensor");
  _tensorNormalize->output("tensor")       >> _tensorToPool->input("tensor");
  _tensorToPool->output("pool")            >> _tensorflowPredict->input("poolIn");
  _tensorflowPredict->output("poolOut")    >> _poolToTensor->input("pool");
  _poolToTensor->output("tensor")          >> _tensorToVectorReal->input("tensor");

  attach(_tensorToVectorReal->output("frame"), _predictions);

  _network = new scheduler::Network(_frameCutter);
}


void TensorflowPredictCREPE::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


TensorflowPredictCREPE::~TensorflowPredictCREPE() {
  clearAlgos();
}


void TensorflowPredictCREPE::reset() {
  AlgorithmComposite::reset();
}


void TensorflowPredictCREPE::configure() {
  if (_configured) {
    clearAlgos();
  }

  createInnerNetwork();

  float hopSize = parameter("hopSize").toFloat();  // hop size in milliseconds
  int batchSize = parameter("batchSize").toInt();

  int hopSizeFrames = int(_sampleRate * hopSize / 1000.0);
  vector<int> inputShape({batchSize, 1, 1, _frameSize});

  // Configure the frameCutter to discard the last frame if it
  // is uncompleted to mimic CREPE's behavior. The main goal is
  // to simplify testing but there is not much practical implication.
  _frameCutter->configure("frameSize", _frameSize,
                          "hopSize", hopSizeFrames,
                          "validFrameThresholdRatio", 0.5,
                          "silentFrames", "keep");

  _vectorRealToTensor->configure("shape", inputShape,
                                 "lastPatchMode", "discard",
                                 "patchHopSize", 0);

  _tensorNormalize->configure("scaler", "standard",
                              "axis", 0,
                              "skipConstantSlices", false);

  _configured = true;

  string input = parameter("input").toString();
  string output = parameter("output").toString();

  _tensorToPool->configure("namespace", input);

  _poolToTensor->configure("namespace", output);


  string graphFilename = parameter("graphFilename").toString();
  string savedModel = parameter("savedModel").toString();

  _tensorflowPredict->configure("graphFilename", graphFilename,
                                "savedModel", savedModel,
                                "inputs", vector<string>({input}),
                                "outputs", vector<string>({output}));
}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* TensorflowPredictCREPE::name = "TensorflowPredictCREPE";
const char* TensorflowPredictCREPE::category = "Machine Learning";
const char* TensorflowPredictCREPE::description = DOC(
  "This algorithm generates activations of monophonic audio signals using CREPE models.\n"
  "\n"
  "`input` and `output` are the input and output node names in the neural network and are "
  "defaulted to the names of the official models. `hopSize` allows to change the pitch "
  "estimation rate. `batchSize` controls how many pitch timestamps to process in parallel. "
  "By default it processes everything at the end of the audio stream, but it can be set to "
  "process batches periodically for online applications.\n"
  "\n"
  "The recommended pipeline is as follows::\n"
  "\n"
  "  MonoLoader(sampleRate=16000) >> TensorflowPredictCREPE()\n"
  "\n"
  "Notes:\n"
  "This algorithm does not make any check on the input model so it is "
  "the user's responsibility to make sure it is a valid one.\n"
  "The required sample rate of input signal is 16 KHz. "
  "Other sample rates will lead to an incorrect behavior.\n"
  "\n"
  "References:\n"
  "\n"
  "1. CREPE: A Convolutional Representation for Pitch Estimation. "
  "Jong Wook Kim, Justin Salamon, Peter Li, Juan Pablo Bello. "
  "Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal "
  "Processing (ICASSP), 2018.\n"
  "\n"
  "2. Original models and code at https://github.com/marl/crepe/\n"
  "\n"
  "3. Supported models at https://essentia.upf.edu/models/\n\n");


TensorflowPredictCREPE::TensorflowPredictCREPE() {
    declareInput(_signal, "signal", "the input audio signal sampled at 16 kHz");
    declareOutput(_predictions, "predictions", "the output values from the model node named after `output`");

    createInnerNetwork();
  }


TensorflowPredictCREPE::~TensorflowPredictCREPE() {
  delete _network;
}


void TensorflowPredictCREPE::createInnerNetwork() {
  _tensorflowPredictCREPE = streaming::AlgorithmFactory::create("TensorflowPredictCREPE");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput >> _tensorflowPredictCREPE->input("signal");
  _tensorflowPredictCREPE->output("predictions") >> PC(_pool, "predictions");

  _network = new scheduler::Network(_vectorInput);
}


void TensorflowPredictCREPE::configure() {
  _tensorflowPredictCREPE->configure(INHERIT("graphFilename"),
                                     INHERIT("savedModel"),
                                     INHERIT("input"),
                                     INHERIT("output"),
                                     INHERIT("hopSize"),
                                     INHERIT("batchSize"));
}


void TensorflowPredictCREPE::compute() {
  const vector<Real>& signal = _signal.get();
  vector<vector<Real> >& predictions = _predictions.get();

  if (!signal.size()) {
    throw EssentiaException("TensorflowPredictCREPE: empty input signal");
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


void TensorflowPredictCREPE::reset() {
  _network->reset();
  _pool.remove("predictions");
}

} // namespace standard
} // namespace essentia
