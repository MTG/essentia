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

#include "tempotapticks.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* TempoTapTicks::name = "TempoTapTicks";
const char* TempoTapTicks::description = DOC("This algorithm builds the list of ticks from the period and phase candidates given by the TempoTap algorithm.\n"
"\n"
"Quality: outdated (use TempoTapDegara instead)\n"
"\n"
"References:\n"
"  [1] F. Gouyon, \"A computational approach to rhythm description: Audio\n"    
"  features for the computation of rhythm periodicity functions and their use\n"
"  in tempo induction and music content processing,\" UPF, Barcelona, Spain,\n" 
"  2005.\n");


void TempoTapTicks::configure() {
  _frameHop = parameter("frameHop").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _frameTime = parameter("hopSize").toInt() / _sampleRate;

  // FIXME: MAGIC NUMBER. Shouldn't this be defined by the user?
  // before there used to be a parameter called continuityTolerance,
  // but was not used at all.
  _periodTolerance = 2.;
  _phaseTolerance = 2.;

  reset();
}

void TempoTapTicks::reset() {
  _nframes = 0;
  _nextPhase = -1;
}

void TempoTapTicks::compute() {
  _nframes++;
  vector<Real>& matchingPeriods = _matchingPeriods.get();

  const vector<Real>& periods = _periods.get();
  const vector<Real>& phases = _phases.get();
  vector<Real>& ticks = _ticks.get();

  // tempotapticks needs to be synchronised with tempotap, which only produces output (periods and phases) after frameHop frames.

  // before this modification, periods_copy was an exact copy of periods.
  // now only periods and phases which are strictly positive are kept, this has
  // improved the detection on short audio (see trac ticket #110)
  vector<Real> periods_copy;
  periods_copy.reserve(periods.size());
  vector<Real> phases_copy;
  phases_copy.reserve(phases.size());

  for (int i=0; i<(int)periods.size(); i++) {
    if (periods[i] != 0) {
      periods_copy.push_back(periods[i]);
      phases_copy.push_back(phases[i]);
    }
  }
  if (periods_copy.empty() || phases_copy.empty()) {
    matchingPeriods.clear();
    ticks.clear();
    return;
  }

  // FIXME: Why do we need copies?

  vector<Real> countedBins;
  //vector<Real>& bpmCandidates = _bpmCandidates.get();
  //Real& bpmConfidence = _bpmConfidence.get();
  // rough estimate of the phase as the median of all candidates
  Real phase     = 0;
  Real curperiod = 0, closestPeriod = 0, closestPhase = 0;

  //if (periods.size() > 0) periods_copy = periods;
  //if (phases.size() > 0) {
    //phases_copy = phases;
  //FIXME: WARNING: MAGIC NUMBER
  // Don't understand what the hell this loop is for???
  // left untouched in case sth breaks
    if (phases_copy.size() > 5) {
      for (int i = 0; i < 4; i++) {
        phases_copy.push_back(phases[4]);
        phases_copy.push_back(phases[3]);
        phases_copy.push_back(phases[2]);
      }
    }
  //}

  if (periods_copy.size() > 0) {
    for (int i = 0; i < int(periods_copy.size()); ++i) {
      // FIXME:
      // why do we need to divide by 2 and later multiply by 2?
      // may this be cause the size produced by bincounts could be quite large?
      periods_copy[i] /= 2.;
    }
    bincount(periods_copy, countedBins); // counts the number of occurrences
    closestPeriod = argmax(countedBins)*2.;// so the most likely period will be the max
    for (int i = 0; i < int(periods_copy.size()); ++i) {
      periods_copy[i] *= 2.; //multiply by 2. why did we divide by 2 then?
      if (abs(closestPeriod - periods_copy[i]) < _periodTolerance) {
        matchingPeriods.push_back(periods_copy[i]);
      }
    }
    if (matchingPeriods.size() < 1) {
      // something odd happened
      curperiod = closestPeriod;
    }
    else {
      curperiod = mean(matchingPeriods);
    }
  }

  if (phases_copy.size() > 0) {
    vector<Real> matchingPhase;
    for (int i=0; i < int(phases_copy.size()); ++i) {
      phases_copy[i] /= 2.;
    }
    bincount(phases_copy, countedBins);
    closestPhase = argmax(countedBins)*2.;
    for (int i = 0; i < int(phases_copy.size()); i++) {
      phases_copy[i] *= 2.;
      if (abs(closestPhase - phases_copy[i]) < _phaseTolerance) {
        matchingPhase.push_back(phases_copy[i]);
      }
    }
    if (matchingPhase.size() < 1) {
      // something odd happened
      phase = closestPhase;
    }
    else {
      phase = mean(matchingPhase);
    }
  }

  ticks.clear();

  if (curperiod > 0) {
    while (phase < _frameHop) {
      ticks.push_back((_nframes - _frameHop + phase) * _frameTime);
      phase += curperiod;
    }
    _nextPhase = (int)round(phase) % _frameHop;
    while (_nextPhase > curperiod) {
      _nextPhase -= (int)round(curperiod);
    }
  }
}
