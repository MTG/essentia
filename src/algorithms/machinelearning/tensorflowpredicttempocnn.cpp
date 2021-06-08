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

#include "tensorflowpredicttempocnn.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* TensorflowPredictTempoCNN::name = essentia::standard::TensorflowPredictTempoCNN::name;
const char* TensorflowPredictTempoCNN::category = essentia::standard::TensorflowPredictTempoCNN::category;
const char* TensorflowPredictTempoCNN::description = essentia::standard::TensorflowPredictTempoCNN::description;


TensorflowPredictTempoCNN::TensorflowPredictTempoCNN() : AlgorithmComposite(),
    _frameCutter(0), _tensorflowInputTempoCNN(0), _vectorRealToTensor(0), _tensorNormalize(0),
    _tensorTranspose(0), _tensorToPool(0), _tensorflowPredict(0), _poolToTensor(0),
    _tensorToVectorReal(0), _configured(false) {

  declareInput(_signal, 4096, "signal", "the input audio signal sampled at 11025 Hz");
  declareOutput(_predictions, 0, "predictions", "the output values from the model node named after `output`");
}


void TensorflowPredictTempoCNN::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter             = factory.create("FrameCutter");
  _tensorflowInputTempoCNN = factory.create("TensorflowInputTempoCNN");
  _vectorRealToTensor      = factory.create("VectorRealToTensor");
  _tensorNormalize         = factory.create("TensorNormalize");
  _tensorTranspose         = factory.create("TensorTranspose");
  _tensorToPool            = factory.create("TensorToPool");
  _tensorflowPredict       = factory.create("TensorflowPredict");
  _poolToTensor            = factory.create("PoolToTensor");
  _tensorToVectorReal      = factory.create("TensorToVectorReal");

  _tensorflowInputTempoCNN->output("bands").setBufferType(BufferUsage::forMultipleFrames);

  _signal                                   >> _frameCutter->input("signal");
  _frameCutter->output("frame")             >> _tensorflowInputTempoCNN->input("frame");
  _tensorflowInputTempoCNN->output("bands") >> _vectorRealToTensor->input("frame");
  _vectorRealToTensor->output("tensor")     >> _tensorNormalize->input("tensor");
  _tensorNormalize->output("tensor")        >> _tensorTranspose->input("tensor");
  _tensorTranspose->output("tensor")        >> _tensorToPool->input("tensor");
  _tensorToPool->output("pool")             >> _tensorflowPredict->input("poolIn");
  _tensorflowPredict->output("poolOut")     >> _poolToTensor->input("pool");
  _poolToTensor->output("tensor")           >> _tensorToVectorReal->input("tensor");

  attach(_tensorToVectorReal->output("frame"), _predictions);

  _network = new scheduler::Network(_frameCutter);
}


void TensorflowPredictTempoCNN::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


TensorflowPredictTempoCNN::~TensorflowPredictTempoCNN() {
  clearAlgos();
}


void TensorflowPredictTempoCNN::reset() {
  AlgorithmComposite::reset();
}


void TensorflowPredictTempoCNN::configure() {
  if (_configured) {
    clearAlgos();
  }

  createInnerNetwork();

  int patchHopSize = parameter("patchHopSize").toInt();
  string lastPatchMode = parameter("lastPatchMode").toString();
  int batchSize = parameter("batchSize").toInt();

  if (batchSize == 0) {
    throw EssentiaException("TensorflowPredictTempoCNN: 0 is not a valid `batchSize` value.");
  }

  // Hardcoded parameters matching the training setup:
  // https://github.com/hendriks73/tempo-cnn/blob/master/tempocnn/feature.py
  int frameSize = 1024;
  int hopSize = 512;
  int patchSize = 256;
  int numberBands = 40;
  vector<int> inputShape({batchSize, 1, patchSize, numberBands});
  string scaler = "standard";

  // Hendrik's models expect data shaped as {Batch, Mels, Time, Channel}.
  vector<int> permutation({0, 3, 2, 1});

  _frameCutter->configure("frameSize", frameSize, "hopSize", hopSize);

  _vectorRealToTensor->configure("shape", inputShape,
                                 "lastPatchMode", lastPatchMode,
                                 "patchHopSize", patchHopSize);

  _tensorNormalize->configure("scaler", scaler);
  
  _tensorTranspose->configure("permutation", permutation);

  _configured = true;

  string input = parameter("input").toString();
  string output = parameter("output").toString();

  _tensorToPool->configure("namespace", input);

  _poolToTensor->configure("namespace", output);

  Parameter graphFilenameParam = parameter("graphFilename");
  // if no file has been specified, do not do anything else
  if (!graphFilenameParam.isConfigured()) return;

  string graphFilename = parameter("graphFilename").toString();

  _tensorflowPredict->configure("graphFilename", graphFilename,
                                "squeeze", false,
                                "inputs", vector<string>({input}),
                                "outputs", vector<string>({output}));
}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* TensorflowPredictTempoCNN::name = "TensorflowPredictTempoCNN";
const char* TensorflowPredictTempoCNN::category = "Machine Learning";
const char* TensorflowPredictTempoCNN::description = DOC(
  "This algorithm makes predictions using TempoCNN-based models.\n"
  "\n"
  "Internally, it uses TensorflowInputTempoCNN for the input feature "
  "extraction (mel bands). It feeds the model with patches of 256 mel bands "
  "frames and jumps a constant amount of frames determined by `patchHopSize`.\n"
  "\n"
  "With the `batchSize` parameter set to -1 the patches are stored to run a "
  "single TensorFlow session at the end of the stream. This allows to take "
  "advantage of parallelization when GPUs are available, but at the same time "
  "it can be memory exhausting for long files.\n"
  "\n"
  "The recommended pipeline is as follows::\n"
  "\n"
  "  MonoLoader(sampleRate=11025) >> TensorflowPredictTempoCNN\n"
  "\n"
  "Note: This algorithm does not make any check on the input model so it is "
  "the user's responsibility to make sure it is a valid one.\n"
  "\n"
  "References:\n"
  "\n"
  "1. Hendrik Schreiber, Meinard Müller, A Single-Step Approach to Musical "
  "Tempo Estimation Using a Convolutional Neural Network Proceedings of the "
  "19th International Society for Music Information Retrieval Conference "
  "(ISMIR), Paris, France, Sept. 2018.\n\n"
  "2. Hendrik Schreiber, Meinard Müller, Musical Tempo and Key Estimation "
  "using Convolutional Neural Networks with Directional Filters Proceedings of "
  "the Sound and Music Computing Conference (SMC), Málaga, Spain, 2019.\n\n"
  "3. Original models and code at https://github.com/hendriks73/tempo-cnn\n\n"
  "4. Supported models at https://essentia.upf.edu/models/\n\n");


TensorflowPredictTempoCNN::TensorflowPredictTempoCNN() {
    declareInput(_signal, "signal", "the input audio signal sampled at 11025 Hz");
    declareOutput(_predictions, "predictions", "the output values from the model node named after `output`");

    createInnerNetwork();
  }


TensorflowPredictTempoCNN::~TensorflowPredictTempoCNN() {
  delete _network;
}


void TensorflowPredictTempoCNN::createInnerNetwork() {
  _tensorflowPredictTempoCNN = streaming::AlgorithmFactory::create("TensorflowPredictTempoCNN");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >> _tensorflowPredictTempoCNN->input("signal");
  _tensorflowPredictTempoCNN->output("predictions") >>  PC(_pool, "predictions");

  _network = new scheduler::Network(_vectorInput);
}


void TensorflowPredictTempoCNN::configure() {
  // if no file has been specified, do not do anything
  if (!parameter("graphFilename").isConfigured()) return;
  _tensorflowPredictTempoCNN->configure(INHERIT("graphFilename"),
                                       INHERIT("input"),
                                       INHERIT("output"),
                                       INHERIT("patchHopSize"),
                                       INHERIT("batchSize"),
                                       INHERIT("lastPatchMode"));
}


void TensorflowPredictTempoCNN::compute() {
  const vector<Real>& signal = _signal.get();
  vector<vector<Real> >& predictions = _predictions.get();

  if (!signal.size()) {
    throw EssentiaException("TensorflowPredictTempoCNN: empty input signal");
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


void TensorflowPredictTempoCNN::reset() {
  _network->reset();
  _pool.remove("predictions");
}

} // namespace standard
} // namespace essentia
