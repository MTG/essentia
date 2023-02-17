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

#include "tensorflowpredictfsdsinet.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* TensorflowPredictFSDSINet::name = essentia::standard::TensorflowPredictFSDSINet::name;
const char* TensorflowPredictFSDSINet::category = essentia::standard::TensorflowPredictFSDSINet::category;
const char* TensorflowPredictFSDSINet::description = essentia::standard::TensorflowPredictFSDSINet::description;


TensorflowPredictFSDSINet::TensorflowPredictFSDSINet() : AlgorithmComposite(),
    _frameCutter(0), _tensorflowInputFSDSINet(0), _vectorRealToTensor(0), _tensorToPool(0),
    _tensorTranspose(0), _tensorflowPredict(0), _poolToTensor(0), _tensorToVectorReal(0), _configured(false) {

  declareInput(_signal, 4096, "signal", "the input audio signal sampled at 22.05 kHz");
  declareOutput(_predictions, 0, "predictions", "the output values from the model node named after `output`");
}


void TensorflowPredictFSDSINet::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter            = factory.create("FrameCutter");
  _tensorflowInputFSDSINet = factory.create("TensorflowInputFSDSINet");
  _vectorRealToTensor     = factory.create("VectorRealToTensor");
  _tensorTranspose        = factory.create("TensorTranspose");
  _tensorToPool           = factory.create("TensorToPool");
  _tensorflowPredict      = factory.create("TensorflowPredict");
  _poolToTensor           = factory.create("PoolToTensor");
  _tensorToVectorReal     = factory.create("TensorToVectorReal");

  _tensorflowInputFSDSINet->output("bands").setBufferType(BufferUsage::forMultipleFrames);
  
  _signal                                  >> _frameCutter->input("signal");
  _frameCutter->output("frame")            >> _tensorflowInputFSDSINet->input("frame");
  _tensorflowInputFSDSINet->output("bands") >> _vectorRealToTensor->input("frame");
  _vectorRealToTensor->output("tensor")     >> _tensorTranspose->input("tensor");
  _tensorTranspose-> output("tensor")      >>  _tensorToPool->input("tensor");
  _tensorToPool->output("pool")            >>  _tensorflowPredict->input("poolIn");
  _tensorflowPredict->output("poolOut")    >>  _poolToTensor->input("pool");
  _poolToTensor->output("tensor")          >>  _tensorToVectorReal->input("tensor");

  attach(_tensorToVectorReal->output("frame"), _predictions);

  _network = new scheduler::Network(_frameCutter);
}


void TensorflowPredictFSDSINet::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


TensorflowPredictFSDSINet::~TensorflowPredictFSDSINet() {
  clearAlgos();
}


void TensorflowPredictFSDSINet::reset() {
  AlgorithmComposite::reset();
}


void TensorflowPredictFSDSINet::configure() {
  if (_configured) {
    clearAlgos();
  }

  createInnerNetwork();

  string lastPatchMode = parameter("lastPatchMode").toString();
  int patchHopSize = parameter("patchHopSize").toInt();
  int batchSize = parameter("batchSize").toInt();

  int frameSize = 660;
  int hopSize = 220;
  int numberBands = 96;
  int patchSize = 101;

  vector<int> inputShape({batchSize, 1, patchSize, numberBands});

  _frameCutter->configure("frameSize", frameSize, "hopSize", hopSize);

  _vectorRealToTensor->configure("shape", inputShape,
                                 "lastPatchMode", lastPatchMode,
                                 "patchHopSize", patchHopSize);

  _configured = true;

  string input = parameter("input").toString();
  string output = parameter("output").toString();

  // {Batch, Channel, Time, Freq} --> {Batch, Time, Freq, Channel}
  vector<int> permutation({0, 2, 3, 1});
  _tensorTranspose->configure("permutation", permutation);
  _tensorToPool->configure("namespace", input);
  _poolToTensor->configure("namespace", output);

  string graphFilename = parameter("graphFilename").toString();
  string savedModel = parameter("savedModel").toString();

  _tensorflowPredict->configure("graphFilename", graphFilename,
                                "savedModel", savedModel,
                                "inputs", vector<string>({input}),
                                "outputs", vector<string>({output}),
                                "squeeze", false,
                                "isTrainingName", "");
}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* TensorflowPredictFSDSINet::name = "TensorflowPredictFSDSINet";
const char* TensorflowPredictFSDSINet::category = "Machine Learning";
const char* TensorflowPredictFSDSINet::description = DOC(
  "This algorithm makes predictions using FSD-SINet models.\n"
  "\n"
  "Internally, it uses TensorflowInputFSDSINet for the input feature extraction "
  "(mel bands). It feeds the model with patches of 101 mel-band frames and "
  "jumps a constant amount of frames determined by `patchHopSize`.\n"
  "\n"
  "By setting the `batchSize` parameter to -1 or 0 the patches are stored to run a single "
  "TensorFlow session at the end of the stream. This allows to take advantage "
  "of parallelization when GPUs are available, but at the same time it can be "
  "memory exhausting for long files.\n"
  "\n"
  "The recommended pipeline is as follows::\n"
  "\n"
  "  MonoLoader(sampleRate=22050) >> TensorflowPredictFSDSINet\n"
  "\n"
  "Note: This algorithm does not make any check on the input model so it is "
  "the user's responsibility to make sure it is a valid one.\n"
  "\n"
  "Note: The FSD-SINet models were trained on normalized audio clips. "
  "Clip-level normalization is only implemented in standard mode since in streaming there is no access to the entire audio clip. "
  "In the streaming case, the user is responsible for controlling the dynamic range of the input signal. "
  "Ideally, the signal should be zero-mean (no DC) and normalized to the full dynamic range (-1, 1).\n\n"
  "References:\n"
  "  [1] Fonseca, E., Ferraro, A., & Serra, X. (2021). Improving sound event classification by increasing shift invariance in convolutional neural networks. arXiv preprint arXiv:2107.00623.\n"
  "  [2] https://github.com/edufonseca/shift_sec"
);

TensorflowPredictFSDSINet::TensorflowPredictFSDSINet() : _normalize(true) {
    declareInput(_signal, "signal", "the input audio signal sampled at 22050 Hz");
    declareOutput(_predictions, "predictions", "the output values from the model's node (layer) named after the `output` parameter");

    createInnerNetwork();
  }


TensorflowPredictFSDSINet::~TensorflowPredictFSDSINet() {
  delete _network;
}


void TensorflowPredictFSDSINet::createInnerNetwork() {
  _tensorflowPredictFSDSINet = streaming::AlgorithmFactory::create("TensorflowPredictFSDSINet");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >> _tensorflowPredictFSDSINet->input("signal");
  _tensorflowPredictFSDSINet->output("predictions") >>  PC(_pool, "predictions");

  _network = new scheduler::Network(_vectorInput);
}

void TensorflowPredictFSDSINet::normalizeFSDSINet(vector<Real> &x) {
  Real meanValue = mean(x);
  for (int i=0; i<(int)x.size(); i++) x[i] -= meanValue;

  // Custom headroom used in the original implementation.
  Real headroom = 0.005;
  normalizeAbs(x, headroom);
}

void TensorflowPredictFSDSINet::configure() {
  _normalize = parameter("normalize").toBool();
  _tensorflowPredictFSDSINet->configure(INHERIT("graphFilename"),
                                       INHERIT("savedModel"),
                                       INHERIT("input"),
                                       INHERIT("output"),
                                       INHERIT("patchHopSize"),
                                       INHERIT("lastPatchMode"),
                                       INHERIT("batchSize"));
}


void TensorflowPredictFSDSINet::compute() {
  const vector<Real>* signal = &_signal.get();
  vector<vector<Real> >& predictions = _predictions.get();
  vector<Real> normalizedSignal;

  if (!signal->size()) {
    throw EssentiaException("TensorflowPredictFSDSINet: empty input signal");
  }

  if (_normalize) {
    normalizedSignal = *signal;
    normalizeFSDSINet(normalizedSignal);
    signal = &normalizedSignal;
  }


  _vectorInput->setVector(signal);

  _network->run();

  try {
    predictions = _pool.value<vector<vector<Real> > >("predictions");
  }
  catch (EssentiaException&) {
    predictions.clear();
  }

  reset();
}


void TensorflowPredictFSDSINet::reset() {
  _network->reset();
  _pool.remove("predictions");
}

} // namespace standard
} // namespace essentia
