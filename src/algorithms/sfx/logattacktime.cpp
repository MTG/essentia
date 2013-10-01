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

#include "logattacktime.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* LogAttackTime::name = "LogAttackTime";
const char* LogAttackTime::description = DOC("This algorithm computes the log (base 10) of the attack time of a signal envelope. The attack time is defined as the time duration from when the sound becomes perceptually audible to when it reaches its maximum intensity. By default, the start of the attack is estimated as the point where the signal envelope reaches 20% of its maximum value in order to account for possible noise presence. Also by default, the end of the attack is estimated as as the point where the signal envelope has reached 90% of its maximum value, in order to account for the possibility that the max value occurres after the logAttack, as in trumpet sounds.\n\n"
"With this said, LogAttackTime's input is intended to be fed by the output of the Envelope algorithm. In streaming mode, the RealAccumulator algorithm should be connected between Envelope and LogAttackTime.\n\n"
"Note that startAttackThreshold cannot be greater than stopAttackThreshold and the input signal should not be empty. In any of these cases an exception will be thrown.\n");

void LogAttackTime::configure() {
  _startThreshold = parameter("startAttackThreshold").toReal();
  _stopThreshold = parameter("stopAttackThreshold").toReal();

  if (_startThreshold > _stopThreshold) {
    throw EssentiaException("LogAttackTime: stopAttackThreshold is not greater than startAttackThreshold");
  }
}

void LogAttackTime::compute() {

  const std::vector<Real>& signal = _signal.get();
  Real& logAttackTime = _logAttackTime.get();

  if (signal.empty()) {
    throw EssentiaException("LogAttackTime: logAttackTime not defined for empty input");
  }

  Real maxvalue = *max_element(signal.begin(), signal.end());

  Real startAttack = 0.0;
  Real cutoffStartAttack = Real(maxvalue * _startThreshold);
  Real stopAttack = 0.0;
  Real cutoffStopAttack = Real(maxvalue * _stopThreshold);

  int i = 0;

  for (; i<int(signal.size()); ++i) {
    if (signal[i] >= cutoffStartAttack) {
      startAttack = Real(i);
      break;
    }
  }

  for (; i<int(signal.size()); i++) {
    if (signal[i] >= cutoffStopAttack) {
      stopAttack = Real(i);
      break;
    }
  }

  Real attackTime = (stopAttack - startAttack)/parameter("sampleRate").toReal();

  if (attackTime > 10e-5) {
    logAttackTime = log10(attackTime);
  }
  else{
    logAttackTime = -5;
  }
}
