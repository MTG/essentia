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

#include "tensorflowpredictmaest.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* TensorflowPredictMAEST::name = essentia::standard::TensorflowPredictMAEST::name;
const char* TensorflowPredictMAEST::category = essentia::standard::TensorflowPredictMAEST::category;
const char* TensorflowPredictMAEST::description = essentia::standard::TensorflowPredictMAEST::description;


TensorflowPredictMAEST::TensorflowPredictMAEST() : AlgorithmComposite(),
    _frameCutter(0), _tensorflowInputMusiCNN(0), _shift(0), _scale(0), _vectorRealToTensor(0),
    _tensorToPool(0), _tensorflowPredict(0), _poolToTensor(0), _configured(false) {

  declareInput(_signal, 480000, "signal", "the input audio signal sampled at 16 kHz");
  declareOutput(_predictions, 1, "predictions", "the output values from the model node named after `output`");
}


void TensorflowPredictMAEST::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter            = factory.create("FrameCutter");
  _tensorflowInputMusiCNN = factory.create("TensorflowInputMusiCNN");
  _shift                  = factory.create("UnaryOperator");
  _scale                  = factory.create("UnaryOperator");
  _vectorRealToTensor     = factory.create("VectorRealToTensor");
  _tensorToPool           = factory.create("TensorToPool");
  _tensorflowPredict      = factory.create("TensorflowPredict");
  _poolToTensor           = factory.create("PoolToTensor");

  _shift->output("array").setBufferType(BufferUsage::forMultipleFrames);
  _scale->output("array").setBufferType(BufferUsage::forMultipleFrames);
  _tensorflowInputMusiCNN->output("bands").setBufferType(BufferUsage::forMultipleFrames);

  _signal                                  >> _frameCutter->input("signal");
  _frameCutter->output("frame")            >> _tensorflowInputMusiCNN->input("frame");
  _tensorflowInputMusiCNN->output("bands") >> _shift->input("array");
  _shift->output("array")                  >> _scale->input("array");
  _scale->output("array")                  >> _vectorRealToTensor->input("frame");
  _vectorRealToTensor->output("tensor")    >> _tensorToPool->input("tensor");
  _tensorToPool->output("pool")            >> _tensorflowPredict->input("poolIn");
  _tensorflowPredict->output("poolOut")    >> _poolToTensor->input("pool");

  attach(_poolToTensor->output("tensor"), _predictions);

  _network = new scheduler::Network(_frameCutter);
}


void TensorflowPredictMAEST::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


TensorflowPredictMAEST::~TensorflowPredictMAEST() {
  clearAlgos();
}


void TensorflowPredictMAEST::reset() {
  AlgorithmComposite::reset();
}


void TensorflowPredictMAEST::configure() {
  if (_configured) {
    clearAlgos();
  }

  createInnerNetwork();

  int patchHopSize = parameter("patchHopSize").toInt();
  string lastPatchMode = parameter("lastPatchMode").toString();
  int patchSize = parameter("patchSize").toInt();
  int batchSize = parameter("batchSize").toInt();

  string input = parameter("input").toString();
  string output = parameter("output").toString();
  string isTrainingName = parameter("isTrainingName").toString();

  string graphFilename = parameter("graphFilename").toString();
  string savedModel = parameter("savedModel").toString();


 // Note the small difference between the patchHopSize and the patchSize parameters below.
 // The patchHopSize is set to jump exactly 30, 20, 10, or 5 seconds.
 // The patchSize is the closest number suitable considering the kernel and stride sizes of the
 // Transformer's embedding layer:
 // https://cs231n.github.io/convolutional-networks/#conv

  if (parameter("patchSize").isConfigured()) {
    if (graphFilename.find("discogs-maest-20s-") != std::string::npos) {
      E_INFO("TensorFlowPredictMAEST: The default `patchSize` is not suitable according to the graph filename `" << graphFilename.c_str() << "`. Setting it to 1256, which is adequate for the 20s model.");
      patchSize = 1256;
    } else if (graphFilename.find("discogs-maest-10s-") != std::string::npos) {
      E_INFO("TensorFlowPredictMAEST: The default `patchSize` is not suitable according to the graph filename `" << graphFilename.c_str() << "`. Setting it to 626, which is adequate for the 10s model.");
      patchSize = 626;
    } else if (graphFilename.find("discogs-maest-5s-") != std::string::npos) {
      E_INFO("TensorFlowPredictMAEST: The default `patchSize` is not suitable according to the graph filename `" << graphFilename.c_str() << "`. Setting it to 316, which is adequate for the 5s model.");
      patchSize = 316;
    }
  }

  if (parameter("patchHopSize").isConfigured()) {
    if (graphFilename.find("discogs-maest-20s-") != std::string::npos) {
      E_INFO("TensorFlowPredictMAEST: The default `patchHopSize` is not suitable according to the graph filename `" << graphFilename.c_str() << "`. Setting it to 1250, which is adequate for the 20s model.\n");
      patchHopSize = 1250;
    } else if (graphFilename.find("discogs-maest-10s-") != std::string::npos) {
      E_INFO("TensorFlowPredictMAEST: The default `patchHopSize` is not suitable according to the graph filename `" << graphFilename.c_str() << "`. Setting it to 625, which is adequate for the 10s model.\n");
      patchHopSize = 625;
    } else if (graphFilename.find("discogs-maest-5s-") != std::string::npos) {
      E_INFO("TensorFlowPredictMAEST: The default `patchHopSize` is not suitable according to the graph filename `" << graphFilename.c_str() << "`. Setting it to 313, which is adequate for the 5s model.\n");
      patchHopSize = 313;
    }
  }


  vector<int> inputShape({batchSize, 1, patchSize, _numberBands});

  _frameCutter->configure("frameSize", _frameSize, "hopSize", _hopSize);

  _vectorRealToTensor->configure("shape", inputShape,
                                 "lastPatchMode", lastPatchMode,
                                 "patchHopSize", patchHopSize);

  _shift->configure("shift", -_mean);
  _scale->configure("scale", 1.0 / (_std * 2));

  _configured = true;


  _tensorToPool->configure("namespace", input);

  _poolToTensor->configure("namespace", output);


  _tensorflowPredict->configure("graphFilename", graphFilename,
                                "savedModel", savedModel,
                                "inputs", vector<string>({input}),
                                "outputs", vector<string>({output}),
                                "isTrainingName", isTrainingName);
}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* TensorflowPredictMAEST::name = "TensorflowPredictMAEST";
const char* TensorflowPredictMAEST::category = "Machine Learning";
const char* TensorflowPredictMAEST::description = DOC(
  "This algorithm makes predictions using MAEST-based models.\n"
  "\n"
  "Internally, it uses TensorflowInputMusiCNN for the input feature extraction. "
  "It feeds the model with mel-spectrogram patches and jumps a constant amount "
  "of frames determined by `patchHopSize`.\n"
  "\n"
  "By setting the `batchSize` parameter to -1 or 0 the patches are stored to run a single "
  "TensorFlow session at the end of the stream. This allows to take advantage "
  "of parallelization when GPUs are available, but at the same time it can be "
  "memory exhausting for long files.\n"
  "\n"
  "For the official MAEST models, the algorithm outputs the probabilities for "
  "400 music style labels by default. Additionally, it is possible to retrieve "
  "the output of each attention layer by setting `output=StatefulParitionedCall:n`, "
  "where `n` is the index of the layer (starting from 1).\n"
  "The output from the attention layers should be interpreted as follows:\n"
  "  [batch_index, 1, token_number, embeddings_size]\n"
  "Where the first and second tokens (e.g., [0, 0, :2, :]) correspond to the "
  "CLS and DIST tokens respectively, and the following ones to input signal ( "
  "refer to the original paper for details [1]).\n"

  "\n"
  "The recommended pipeline is as follows::\n"
  "\n"
  "  MonoLoader(sampleRate=16000, resampleQuality=4) >> TensorflowPredictMAEST\n"
  "\n"
  "Note: this algorithm does not make any check on the input model so it is "
  "the user's responsibility to make sure it is a valid one.\n"
  "\n"
  "Note: when `patchHopSize` and `patchSize` are not specified, the algorithm "
  "will parse the `graphFilename` string to try to set appropriate values.\n"
  "\n"
  "References:\n"
  "\n"
  "1. Alonso-Jiménez, P., Serra, X., & Bogdanov, D. (2023). Efficient Supervised "
  "Training of Audio Transformers for Music Representation Learning. In Proceedings "
  "of the 24th International Society for Music Information Retrieval Conference "
  "(ISMIR 2023)\n\n"
  "2. Supported models at https://essentia.upf.edu/models.html#MAEST\n\n");


TensorflowPredictMAEST::TensorflowPredictMAEST() {
    declareInput(_signal, "signal", "the input audio signal sampled at 16 kHz");
    declareOutput(_predictions, "predictions", "the output values from the model node named after `output`");

    createInnerNetwork();
  }


TensorflowPredictMAEST::~TensorflowPredictMAEST() {
  delete _network;
}


void TensorflowPredictMAEST::createInnerNetwork() {
  _tensorflowPredictMAEST = streaming::AlgorithmFactory::create("TensorflowPredictMAEST");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >> _tensorflowPredictMAEST->input("signal");
  _tensorflowPredictMAEST->output("predictions") >>  PC(_pool, "predictions");

  _network = new scheduler::Network(_vectorInput);
}


void TensorflowPredictMAEST::configure() {
  _tensorflowPredictMAEST->configure(INHERIT("graphFilename"),
                                       INHERIT("savedModel"),
                                       INHERIT("input"),
                                       INHERIT("output"),
                                       INHERIT("isTrainingName"),
                                       INHERIT("patchHopSize"),
                                       INHERIT("lastPatchMode"),
                                       INHERIT("patchSize"),
                                       INHERIT("batchSize"));
}


void TensorflowPredictMAEST::compute() {
  const vector<Real>& signal = _signal.get();
  Tensor<Real>& predictions = _predictions.get();

  if (!signal.size()) {
    throw EssentiaException("TensorflowPredictMAEST: empty input signal");
  }

  _vectorInput->setVector(&signal);

  _network->run();

  try {
    vector<Tensor<Real> > predictions_vector = _pool.value<vector<Tensor<Real> > >("predictions");
    predictions = predictions_vector[0];

    for (int i = 1; i < (int)predictions_vector.size(); i++) {
       Tensor<Real> new_predictions = predictions.concatenate(predictions_vector[i], 0).eval();
       predictions = new_predictions;
    }
  }
  catch (EssentiaException&) {
    reset();

    throw EssentiaException("TensorflowPredictMAEST: input signal is too short.");
  }

  reset();
}


void TensorflowPredictMAEST::reset() {
  _network->reset();
  _pool.remove("predictions");
}

} // namespace standard
} // namespace essentia
