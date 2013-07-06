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

#ifndef ESSENTIA_LOGATTACKTIME_H
#define ESSENTIA_LOGATTACKTIME_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class LogAttackTime : public Algorithm {

 private:
  Real _startThreshold, _stopThreshold;

  Input<std::vector<Real> > _signal;
  Output<Real> _logAttackTime;

 public:
  LogAttackTime() {
    declareInput(_signal, "signal", "the input signal envelope (must be non-empty)");
    declareOutput(_logAttackTime, "logAttackTime", "the log (base 10) of the attack time [log10(s)]");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("startAttackThreshold", "the percentage of the input signal envelope at which the starting point of the attack is considered", "[0,1]", 0.2);
    declareParameter("stopAttackThreshold", "the percentage of the input signal envelope at which the ending point of the attack is considered", "[0,1]", 0.9);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class LogAttackTime : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _logAttackTime;

  std::vector<Real> _accu;

 public:
  LogAttackTime() {
    declareAlgorithm("LogAttackTime");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_logAttackTime, TOKEN, "logAttackTime");
  }

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LOGATTACKTIME_H
