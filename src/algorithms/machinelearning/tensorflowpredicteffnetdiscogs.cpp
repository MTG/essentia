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

#include "tensorflowpredicteffnetdiscogs.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* TensorflowPredictEffnetDiscogs::name = essentia::standard::TensorflowPredictEffnetDiscogs::name;
const char* TensorflowPredictEffnetDiscogs::category = essentia::standard::TensorflowPredictEffnetDiscogs::category;
const char* TensorflowPredictEffnetDiscogs::description = essentia::standard::TensorflowPredictEffnetDiscogs::description;


TensorflowPredictEffnetDiscogs::TensorflowPredictEffnetDiscogs() : AlgorithmComposite(),
    _frameCutter(0), _tensorflowInputMusiCNN(0), _vectorRealToTensor(0), _tensorToPool(0),
    _tensorflowPredict(0), _poolToTensor(0), _tensorToVectorReal(0), _configured(false) {

  declareInput(_signal, 4096, "signal", "the input audio signal sampled at 16 kHz");
  declareOutput(_predictions, 0, "predictions", "the output values from the model node named after `output`");
}


void TensorflowPredictEffnetDiscogs::createInnerNetwork() {
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
  _vectorRealToTensor->output("tensor")    >> _tensorToPool->input("tensor");
  _tensorToPool->output("pool")            >> _tensorflowPredict->input("poolIn");
  _tensorflowPredict->output("poolOut")    >> _poolToTensor->input("pool");
  _poolToTensor->output("tensor")          >> _tensorToVectorReal->input("tensor");

  attach(_tensorToVectorReal->output("frame"), _predictions);

  _network = new scheduler::Network(_frameCutter);
}


void TensorflowPredictEffnetDiscogs::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


TensorflowPredictEffnetDiscogs::~TensorflowPredictEffnetDiscogs() {
  clearAlgos();
}


void TensorflowPredictEffnetDiscogs::reset() {
  AlgorithmComposite::reset();
}


void TensorflowPredictEffnetDiscogs::configure() {
  if (_configured) {
    clearAlgos();
  }

  createInnerNetwork();

  int patchHopSize = parameter("patchHopSize").toInt();
  string lastPatchMode = parameter("lastPatchMode").toString();
  int patchSize = parameter("patchSize").toInt();
  int batchSize = parameter("batchSize").toInt();

  if (patchSize == 0) {
    throw EssentiaException("TensorflowPredictEffnetDiscogs: `patchSize` cannot be 0");
  }


  vector<int> inputShape({batchSize, 1, patchSize, _numberBands});

  _frameCutter->configure("frameSize", _frameSize, "hopSize", _hopSize);

  _vectorRealToTensor->configure("shape", inputShape,
                                 "lastPatchMode", lastPatchMode,
                                 "patchHopSize", patchHopSize,
                                 "lastBatchMode", "discard");

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

const char* TensorflowPredictEffnetDiscogs::name = "TensorflowPredictEffnetDiscogs";
const char* TensorflowPredictEffnetDiscogs::category = "Machine Learning";
const char* TensorflowPredictEffnetDiscogs::description = DOC(
  "This algorithm makes predictions using EffnetDiscogs-based models.\n"
  "\n"
  "Internally, it uses TensorflowInputMusiCNN for the input feature extraction "
  "(mel-spectrograms). It feeds the model with patches of 128 frames and "
  "jumps a constant amount of frames determined by `patchHopSize`.\n"
  "\n"
  "By setting the `batchSize` parameter to -1 or 0 the patches are stored to run a single "
  "TensorFlow session at the end of the stream. This allows to take advantage "
  "of parallelization when GPUs are available, but at the same time it can be "
  "memory exhausting for long files. "
  "This option is not supported by some EffnetDiscogs models that require a fixed batch size.\n"
  "\n"
  "The recommended pipeline is as follows::\n"
  "\n"
  "  MonoLoader(sampleRate=16000) >> TensorflowPredictEffnetDiscogs\n"
  "\n"
  "Note: This algorithm does not make any check on the input model so it is "
  "the user's responsibility to make sure it is a valid one.\n"
  "\n"
  "References:\n"
  "\n"
  "1. Supported models at https://essentia.upf.edu/models/\n\n");


TensorflowPredictEffnetDiscogs::TensorflowPredictEffnetDiscogs() {
    declareInput(_signal, "signal", "the input audio signal sampled at 16 kHz");
    declareOutput(_predictions, "predictions", "the output values from the model node named after `output`");

    createInnerNetwork();
  }


TensorflowPredictEffnetDiscogs::~TensorflowPredictEffnetDiscogs() {
  delete _network;
}


void TensorflowPredictEffnetDiscogs::createInnerNetwork() {
  _tensorflowPredictEffnetDiscogs = streaming::AlgorithmFactory::create("TensorflowPredictEffnetDiscogs");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >> _tensorflowPredictEffnetDiscogs->input("signal");
  _tensorflowPredictEffnetDiscogs->output("predictions") >>  PC(_pool, "predictions");

  _network = new scheduler::Network(_vectorInput);
}


void TensorflowPredictEffnetDiscogs::configure() {
  _tensorflowPredictEffnetDiscogs->configure(INHERIT("graphFilename"),
                                             INHERIT("savedModel"),
                                             INHERIT("input"),
                                             INHERIT("output"),
                                             INHERIT("patchHopSize"),
                                             INHERIT("lastPatchMode"),
                                             INHERIT("batchSize"),
                                             INHERIT("patchSize"));

  _patchHopSize = parameter("patchHopSize").toInt();
  _patchSize = parameter("patchSize").toInt();
  _batchSize = parameter("batchSize").toInt();
  _lastPatchMode = parameter("lastPatchMode").toString();
  _lastBatchMode = parameter("lastBatchMode").toString();
}


void TensorflowPredictEffnetDiscogs::compute() {
  const vector<Real>* signal = &_signal.get();
  vector<vector<Real> >& predictions = _predictions.get();

  if (!signal->size()) {
    throw EssentiaException("TensorflowPredictEffnetDiscogs: empty input signal");
  }

  vector<Real> paddedSignal;
  int paddingPatches;
  if (_batchSize > 0) {
    if (_lastBatchMode == "zeros" || _lastBatchMode == "same") {
      // Computes the number of patches required to fill the final batch and makes a
      //  zero-padded copy of the input signal only when needed.
      paddingPatches = padSignal(*signal, paddedSignal);
      if (paddingPatches) signal = &paddedSignal;
    }
  }

  _vectorInput->setVector(signal);

  _network->run();

  try {
    predictions = _pool.value<vector<vector<Real> > >("predictions");
    if (_lastBatchMode == "same") {
      predictions.erase(predictions.end() - paddingPatches, predictions.end());
    }
  }
  catch (EssentiaException&) {
    predictions.clear();
  }

  reset();
}


void TensorflowPredictEffnetDiscogs::reset() {
  _network->reset();
  _pool.remove("predictions");
}

int TensorflowPredictEffnetDiscogs::padSignal(const std::vector<Real> &signal, std::vector<Real> &paddedSignal) {
  int nSamples = signal.size();

  // FrameCutter zero-pads the signal so that the first frame is zero-centered.
  // Consider that to estimate the number of hops.
  Real hops = ((Real)nSamples - (Real)_frameSize / 2.0) / (Real)_hopSize;

  // By default FrameCutter, zero-pads the final samples to form a last frame.
  int nFrames = 1 + ceil(hops);

  // Frames to patches.
  Real patchHops = ((Real)nFrames - (Real)_patchSize) / (Real)_patchHopSize;

  int nPatches;
  if (_lastPatchMode == "repeat") {
    nPatches = 1 + ceil(patchHops);
  } else if (_lastPatchMode == "discard") {
    nPatches = 1 + floor(patchHops);
  } else {
    throw EssentiaException("TensorflowPredictEffnetDiscogs: incorrect `lastPatchMode`");
  }

  // Patches to batches.
  int nBatches = ceil((Real)nPatches / (Real)_batchSize);

  // How many patches to complete last batch?
  int paddingPatches = nBatches * _batchSize - nPatches;

  if (paddingPatches) {
    // How many frames to complete last batch?
    int missingFrames = paddingPatches * _patchHopSize;
    
    // How many samples to complete the last batch?
    int missingSamples = missingFrames * _hopSize;

    paddedSignal = signal;

    vector<Real> padding(missingSamples, 0.0);
    paddedSignal.insert(paddedSignal.end(), padding.begin(), padding.end());
  }

  return paddingPatches;
}

} // namespace standard
} // namespace essentia
