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

#ifndef ESSENTIA_TEMPOTAPTICKS_H
#define ESSENTIA_TEMPOTAPTICKS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class TempoTapTicks : public Algorithm {

 private:
  Input< std::vector<Real> > _periods;
  Input< std::vector<Real> > _phases;
  Output<std::vector<Real> > _ticks;
  Output<std::vector<Real> > _matchingPeriods;
  Real _frameTime;
  Real _sampleRate;
  int _nextPhase;
  int _frameHop;
  int _nframes;
  Real _periodTolerance;
  Real _phaseTolerance;

 public:
  TempoTapTicks() {
    declareInput(_periods, "periods", "tempo period candidates for the current frame, in frames");
    declareInput(_phases, "phases", "tempo ticks phase candidates for the current frame, in frames");
    declareOutput(_ticks, "ticks", "the list of resulting ticks [s]");
    declareOutput(_matchingPeriods, "matchingPeriods", "list of matching periods [s]");
  }

  ~TempoTapTicks() {};

  void declareParameters() {
    declareParameter("frameHop", "number of feature frames separating two evaluations", "(0,inf)", 512);
    declareParameter("hopSize", "number of audio samples per features", "(0,inf)", 256);
    declareParameter("sampleRate", "sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void configure();
  void compute();
  void reset();

  static const char* name;
  static const char* description;

}; // class TempoTapTicks

} // namespace essentia
} // namespace standard

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TempoTapTicks : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _periods;
  Sink<std::vector<Real> > _phases;
  Source<std::vector<Real> > _ticks;
  Source<std::vector<Real> > _matchingPeriods;

 public:
  TempoTapTicks() {
    declareAlgorithm("TempoTapTicks");
    declareInput(_periods, TOKEN, "periods");
    declareInput(_phases, TOKEN, "phases");
    declareOutput(_ticks, TOKEN, "ticks");
    declareOutput(_matchingPeriods, TOKEN, "matchingPeriods");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TEMPOTAPTICKS_H
