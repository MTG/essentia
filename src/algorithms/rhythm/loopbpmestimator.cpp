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


#include "loopbpmestimator.h"
#include "essentiamath.h"
using namespace std;

namespace essentia {
namespace standard {

const char* LoopBpmEstimator::name = "LoopBpmEstimator";
const char* LoopBpmEstimator::category = "Rhythm";
const char* LoopBpmEstimator::description = DOC("This algorithm estimates the BPM of audio loops. It internally uses "
  "PercivalBpmEstimator algorithm to produce a BPM estimate and LoopBpmConfidence to asses the reliability of the estimate. "
  "If the provided estimate is below the given confidenceThreshold, the algorithm outputs a BPM 0.0, otherwise it outputs "
  "the estimated BPM. For more details on the BPM estimation method and the confidence measure please check the used algorithms."
  );


void LoopBpmEstimator::compute() {
  const vector<Real>& signal = _signal.get();
  Real& bpm = _bpm.get();

  Real bpmEstimate;
  _percivalBpmEstimator->input("signal").set(signal);
  _percivalBpmEstimator->output("bpm").set(bpmEstimate);
  _percivalBpmEstimator->compute();
  bpmEstimate = round(bpmEstimate); // Round the output (loops have integer bpm's)

  Real confidence;
  _loopBpmConfidence->input("signal").set(signal);
  _loopBpmConfidence->input("bpmEstimate").set(bpmEstimate);
  _loopBpmConfidence->output("confidence").set(confidence);
  _loopBpmConfidence->compute();

  if (confidence >= parameter("confidenceThreshold").toReal()) {
    bpm = bpmEstimate;
  } else {
    bpm = 0.0;
  }
}

} // namespace standard
} // namespace essentia
