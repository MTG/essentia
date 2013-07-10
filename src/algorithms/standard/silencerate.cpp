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

#include "silencerate.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* SilenceRate::name = "SilenceRate";
const char* SilenceRate::description = DOC("Given a list of thresholds, this algorithm creates a equally-sized list of outputs and returns 1 on a given output whenever the instant power of the input frame is below the given output's respective threshold, and returns 0 otherwise. This is done for each frame with respect to all outputs. In other words, if a given frame's instant power is below several given thresholds, then each of the corresponding outputs will emit a 1."
);

// the standard version has been written not to use the streaming version which
// was prior to standard one, because it results in a much cleaner code.

void SilenceRate::configure() {

  _thresholds = parameter("thresholds").toVectorReal();
  clearOutputs();
  for (int i=0; i<(int)_thresholds.size(); i++) {
    _outputs.push_back(new Output<Real>());
    ostringstream outputName;
    outputName << "threshold_" << i;
    ostringstream thresholdIndex;
    thresholdIndex << i;
    declareOutput(*_outputs.back(), outputName.str(),
                  "the silence rate for threshold #" + thresholdIndex.str());
  }
}

void SilenceRate::clearOutputs() {
  for (int i=0; i<int(_outputs.size()); ++i) delete _outputs[i];
  _outputs.clear();
}

void SilenceRate::compute() {
  const vector<Real>& frame = _frame.get();

  Real power = instantPower(frame);

  for (int i=0; i<int(_outputs.size()); ++i) {
    Real& output = _outputs[i]->get();
    output = power < _thresholds[i]? 1.0 : 0.0;
  }
}

} // namespace standard
} // namespace essentia

namespace essentia {
namespace streaming {

const char* SilenceRate::name = "SilenceRate";
const char* SilenceRate::description = DOC("Given a list of thresholds, this algorithm creates a equally-sized list of outputs and returns 1 on a given output whenever the instant power of the input frame is below the given output's respective threshold, and returns 0 otherwise. This is done for each frame with respect to all outputs. In other words, if a given frame's instant power is below several given thresholds, then each of the corresponding outputs will emit a 1."
);

void SilenceRate::clearOutputs() {
  for (int i=0; i<(int)_outputs.size(); i++) delete _outputs[i];
  _outputs.clear();
}

void SilenceRate::configure() {
  _thresholds = parameter("thresholds").toVectorReal();

  clearOutputs();
  for (int i=0; i<int(_thresholds.size()); ++i) {
    _outputs.push_back(new Source<Real>());
    ostringstream outputName;
    outputName << "threshold_" << i;
    ostringstream thresholdIndex;
    thresholdIndex << i;
    declareOutput(*_outputs.back(), 1, outputName.str(),
                  "the silence rate for threshold #" + thresholdIndex.str());
  }
}


AlgorithmStatus SilenceRate::process() {
  EXEC_DEBUG("process()");

  AlgorithmStatus status = acquireData();

  if (status != OK) return status;

  const vector<Real>& frame = _frame.firstToken();

  if (frame.empty()) {
    throw EssentiaException("SilenceRate: a given input frame was empty, "
                            "cannot compute the power of an empty frame.");
  }

  Real power = instantPower(frame);

  for (int i=0; i<(int)_outputs.size(); i++) {
    Real& output = _outputs[i]->firstToken();
    output = power < _thresholds[i]? 1.0 : 0.0;
  }

  releaseData();

  return OK;
}


} // namespace streaming
} //namespace essentia
