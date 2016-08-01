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

#include "evaluatepulsetrains.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;

namespace essentia {
namespace standard {

const char* EvaluatePulseTrains::name = "EvaluatePulseTrains";
const char* EvaluatePulseTrains::description = DOC("TODO add description\n");

void EvaluatePulseTrains::configure() {
}

void EvaluatePulseTrains::compute() {
  const vector<Real>& oss = _oss.get();
  const vector<Real>& peakPositions = _peakPositions.get();
  Real& lag = _lag.get();

  // TODO: Correlate OSS signal with ideal expected pulse trains for different peak positions
  // Return the best correlate

  // CURRENT FAKE IMPLEMENTATION: return first peak position
  lag = peakPositions[0];
}

} // namespace standard
} // namespace essentia
