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

#include "tensorflowpredict2d.h"
#include <essentia/utils/tnt/tnt2vector.h>

using namespace std;
using namespace TNT;

namespace essentia {
namespace streaming {

const char* TensorflowPredict2D::name = essentia::standard::TensorflowPredict2D::name;
const char* TensorflowPredict2D::category = essentia::standard::TensorflowPredict2D::category;
const char* TensorflowPredict2D::description = essentia::standard::TensorflowPredict2D::description;


TensorflowPredict2D::TensorflowPredict2D() : AlgorithmComposite(),
    _vectorRealToTensor(0), _tensorToPool(0), _tensorflowPredict(0), _poolToTensor(0),
    _tensorToVectorReal(0), _configured(false) {

  declareInput(_features, 4096, "features", "the input features");
  declareOutput(_predictions, 0, "predictions", "the output predictions from the node named after `output`");
}


void TensorflowPredict2D::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _vectorRealToTensor     = factory.create("VectorRealToTensor");
  _tensorToPool           = factory.create("TensorToPool");
  _tensorflowPredict      = factory.create("TensorflowPredict");
  _poolToTensor           = factory.create("PoolToTensor");
  _tensorToVectorReal     = factory.create("TensorToVectorReal");

  // _tensorflowInput2D->output("bands").setBufferType(BufferUsage::forMultipleFrames);

  _features >> _vectorRealToTensor->input("frame");
  _vectorRealToTensor->output("tensor") >> _tensorToPool->input("tensor");
  _tensorToPool->output("pool") >> _tensorflowPredict->input("poolIn");
  _tensorflowPredict->output("poolOut") >> _poolToTensor->input("pool");
  _poolToTensor->output("tensor") >> _tensorToVectorReal->input("tensor");

  attach(_tensorToVectorReal->output("frame"), _predictions);

  _network = new scheduler::Network(_vectorRealToTensor);
}


void TensorflowPredict2D::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


TensorflowPredict2D::~TensorflowPredict2D() {
  clearAlgos();
}


void TensorflowPredict2D::reset() {
  AlgorithmComposite::reset();
}


void TensorflowPredict2D::configure() {
  if (_configured) {
    clearAlgos();
  }

  createInnerNetwork();

  int patchHopSize = parameter("patchHopSize").toInt();
  string lastPatchMode = parameter("lastPatchMode").toString();
  bool accumulate = parameter("accumulate").toBool();
  int patchSize = parameter("patchSize").toInt();
  int batchSize = parameter("batchSize").toInt();
  int dimensions = parameter("dimensions").toInt();

  if (accumulate) batchSize = -1;

  vector<int> inputShape({batchSize, 1, patchSize, dimensions});

  _vectorRealToTensor->configure("shape", inputShape,
                                 "lastPatchMode", lastPatchMode,
                                 "patchHopSize", patchHopSize);

  _configured = true;

  string input = parameter("input").toString();
  string output = parameter("output").toString();
  string isTrainingName = parameter("isTrainingName").toString();

  _tensorToPool->configure("namespace", input);

  _poolToTensor->configure("namespace", output);

  string graphFilename = parameter("graphFilename").toString();
  string savedModel = parameter("savedModel").toString();

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

const char* TensorflowPredict2D::name = "TensorflowPredict2D";
const char* TensorflowPredict2D::category = "Machine Learning";
const char* TensorflowPredict2D::description = DOC(
  "This algorithm makes predictions using models expecting 2D representations.\n"
  "\n"
  "It expects an input feature matrix with shape (timestamps, dimensions) "
  "and processes it sequentially along the time axis. "
  "Internally, the algorithm generates tensors with shape (batchSize, 1, patchSize, dimensions) "
  "so that the input can be processed by TensorflowPredict. "
  "`patchSize` is the number of timestamps that the model expects, `batchSize` is the number of "
  "patches to be fed to the model at once (most useful when a GPU is available), and `patchHopSize` "
  "is the number of timestamps that separate the beginning of adjacent patches.\n"
  "\n"
  "By setting the `batchSize` parameter to -1 or 0 the patches are stored to run a single "
  "TensorFlow session at the end of the stream. This allows taking advantage "
  "of parallelization when GPUs are available, but at the same time, it can be "
  "memory exhausting for long files.\n"
  "\n"
  "A possible pipeline is as follows::\n"
  "\n"
  "  MonoLoader(sampleRate=16000) >> TensorflowPredictMusiCNN(graphFilename='embedding_model.pb') >> TensorflowPredict2D(graphFilename='classification_model.pb')\n"
  "\n"
  "Note: This algorithm does not make any check on the input model so it is "
  "the user's responsibility to make sure it is a valid one.\n"
  "\n"
  "Note 2: In standard mode, the `dimensions` parameter is overridden with the shape of the input data. "
  "However, in streaming mode, the user is responsible for setting `dimensions` to the adequate value. "
  "Otherwise, an exception is thrown.\n"
  "\n"
  "References:\n"
  "\n"
  "1. Supported models at https://essentia.upf.edu/models/\n\n");


TensorflowPredict2D::TensorflowPredict2D() : _dimensions(0) {
    declareInput(_features, "signal", "the input features");
    declareOutput(_predictions, "predictions", "the output predictions from the node named after `output`");

    createInnerNetwork();
  }


TensorflowPredict2D::~TensorflowPredict2D() {
  delete _network;
}


void TensorflowPredict2D::createInnerNetwork() {
  _tensorflowPredict2D = streaming::AlgorithmFactory::create("TensorflowPredict2D");
  _vectorVectorInput = new streaming::VectorInput<vector<Real> >();

  *_vectorVectorInput >> _tensorflowPredict2D->input("features");
  _tensorflowPredict2D->output("predictions") >>  PC(_pool, "predictions");

  _network = new scheduler::Network(_vectorVectorInput);
}


void TensorflowPredict2D::configure() {
  if (!_dimensions) _dimensions = parameter("dimensions").toInt();

  _tensorflowPredict2D->configure(INHERIT("graphFilename"),
                                       INHERIT("savedModel"),
                                       INHERIT("input"),
                                       INHERIT("output"),
                                       INHERIT("isTrainingName"),
                                       INHERIT("patchHopSize"),
                                       INHERIT("accumulate"),
                                       INHERIT("lastPatchMode"),
                                       INHERIT("patchSize"),
                                       INHERIT("batchSize"),
                                       "dimensions", _dimensions);
}


void TensorflowPredict2D::compute() {
  const Array2D<Real>& features = _features.get();
  vector<vector<Real> >& predictions = _predictions.get();

  if (features.dim2() != _dimensions) {
    _dimensions = features.dim2();
    configure();
  }


  if (!features.dim1()) {
    throw EssentiaException("TensorflowPredict2D: empty input signal");
  }
  vector<vector<Real> > featuresVector = array2DToVecvec(features);
  _vectorVectorInput->setVector(&featuresVector);

  _network->run();

  try {
    predictions = _pool.value<vector<vector<Real> > >("predictions");
  }
  catch (EssentiaException&) {
    predictions.clear();
  }

  reset();
}


void TensorflowPredict2D::reset() {
  _network->reset();
  _pool.remove("predictions");
}

} // namespace standard
} // namespace essentia
