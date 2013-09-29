/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

#include "streamingalgorithmcomposite.h"
#include "algorithms/poolstorage.h"
#include "../scheduler/network.h"
#include "../scheduler/graphutils.h"
using namespace std;

namespace essentia {
namespace streaming {


void AlgorithmComposite::declareInput(SinkBase& sink, const string& name, const string& desc) {
  Algorithm::declareInput(sink, name, desc);
}

void AlgorithmComposite::declareOutput(SourceBase& source, const string& name, const string& desc) {
  Algorithm::declareOutput(source, name, desc);
}


vector<ProcessStep> AlgorithmComposite::processOrder() {
  _processOrder.clear();
  declareProcessOrder();

  if (_processOrder.empty()) {
    throw EssentiaException("The process order for composite algorithm '", name(), "' is empty; please define one.");
  }

  return _processOrder;
}

void AlgorithmComposite::declareProcessStep(const ProcessStep& step) {
  _processOrder.push_back(step);
}


// reset the given algorithm, and if it is a PoolStorage instance, also remove
// the descriptor it was storing in the pool
void resetAlgorithmAndClearPool(Algorithm* algo) {
  algo->reset();
  PoolStorageBase* pstorage = dynamic_cast<PoolStorageBase*>(algo);
  if (pstorage) {
    pstorage->pool()->remove(pstorage->descriptorName());
  }
}

void AlgorithmComposite::reset() {
  E_DEBUG(EAlgorithm, "Streaming: " << name() << " AlgorithmComposite::reset()");

  E_DEBUG_INDENT;
  Algorithm::reset();
  E_DEBUG_OUTDENT;

  E_DEBUG(EAlgorithm, "Streaming: " << name() << " AlgorithmComposite::reset(), resetting inner algorithms");

  // find all algos used by this composite in its process order and reset them
  E_DEBUG_INDENT;
  vector<ProcessStep> porder = processOrder();
  for (int i=0; i<(int)porder.size(); i++) {
    ProcessStep& pstep = porder[i];

    if (pstep.type() == "chain") {
      vector<Algorithm*> algos = scheduler::Network::innerVisibleAlgorithms(pstep.algorithm());
      for (int i=0; i<(int)algos.size(); i++) {
        resetAlgorithmAndClearPool(algos[i]);
      }
    }
    else if (pstep.type() == "single") {
      if (pstep.algorithm() != this) resetAlgorithmAndClearPool(pstep.algorithm());
    }
    else throw EssentiaException("Invalid process step when trying to reset AlgorithmComposite ", name());
  }
  E_DEBUG_OUTDENT;

  E_DEBUG(EAlgorithm, "Streaming: " << name() << " AlgorithmComposite::reset() ok!");
}

} // namespace streaming
} // namespace essentia
