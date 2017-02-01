/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "musicextractorsvm.h"
#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace standard {

const char* MusicExtractorSVM::name = "MusicExtractorSVM";
const char* MusicExtractorSVM::category = "Extractors";
const char* MusicExtractorSVM::description = DOC("This algorithms computes SVM predictions given a pool with aggregated descriptor values computed by MusicExtractor.");

// TODO explain better (reuse music extractor svm documentation)


MusicExtractorSVM::MusicExtractorSVM() {
  declareInput(_inputPool, "pool", "aggregated pool of extracted values");
  declareOutput(_outputPool, "pool", "pool with classification results (resulting from the transformation of the gaia point)");
}

MusicExtractorSVM::~MusicExtractorSVM() {
  for (int i = 0; i < (int)_svms.size(); i++) {
    if (_svms[i]) {
      delete _svms[i];
    }
  }
}

void MusicExtractorSVM::reset() {}

void MusicExtractorSVM::configure() {

  if (parameter("svms").isConfigured()) { 
    vector<string> svmModels = parameter("svms").toVectorString();

    for (int i=0; i<(int) svmModels.size(); i++) {
      E_INFO("MusicExtractorSVM: adding SVM model " << svmModels[i]);
      Algorithm* svm = AlgorithmFactory::create("GaiaTransform", "history", svmModels[i]);
      _svms.push_back(svm);
    }
  }
  else {
    E_INFO("MusicExtractorSVM: no classifier models were configured by default");
  }
}


void MusicExtractorSVM::compute() {

  const Pool& inputPool = _inputPool.get();
  Pool& outputPool = _outputPool.get();

  for (int i = 0; i < (int)_svms.size(); i++) {
    _svms[i]->input("pool").set(inputPool);
    _svms[i]->output("pool").set(outputPool);
    _svms[i]->compute();
  }
}

} // namespace standard
} // namespace essentia
