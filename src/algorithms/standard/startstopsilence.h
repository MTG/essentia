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

#ifndef STARTSTOPSILENCE_H
#define STARTSTOPSILENCE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class StartStopSilence : public Algorithm {

 private:
  Input<std::vector<Real> > _frame;
  Output<int> _startSilenceSource;
  Output<int> _stopSilenceSource;

  int _startSilence;
  int _stopSilence;
  int _nFrame;
  bool _wasSilent;
  Real _threshold;

 public:
  StartStopSilence() {
    declareInput(_frame, "frame", "the input audio frames");
    declareOutput(_startSilenceSource, "startFrame", "number of the first non-silent frame");
    declareOutput(_stopSilenceSource, "stopFrame", "number of the last non-silent frame");
    reset();
  }


  void declareParameters() {
    declareParameter("threshold", "the threshold below which average energy is defined as silence [dB]", "(-inf,0])", -60);
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class StartStopSilence : public Algorithm {

 protected:
  int _startSilence;
  int _stopSilence;
  int _nFrame;
  Real _threshold;

  Source<int> _startSilenceSource;
  Source<int> _stopSilenceSource;
  Sink<std::vector<Real> > _frame;

 public:
  StartStopSilence() {
    declareInput(_frame, 1, "frame", "the input audio frames");
    declareOutput(_startSilenceSource, 0, "startFrame", "number of the first non-silent frame");
    declareOutput(_stopSilenceSource, 0, "stopFrame", "number of the last non-silent frame");
  }

  void declareParameters() {
    declareParameter("threshold", "the threshold below which average energy is defined as silence [dB]", "(-inf,0])", -60);
  }

  void configure();
  AlgorithmStatus process();
  void reset() {
    // TODO: why Algorithm::reset() ?
    Algorithm::reset();
    _startSilence = 0;
    _stopSilence = 0;
    _nFrame = 0;
  }

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // STARTSTOPSILENCE_H
