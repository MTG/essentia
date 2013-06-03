/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "streamingalgorithmcomposite.h"
#include "network.h"
#include "graphutils.h"
#include "poolstorage.h"
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
