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

#include "bpmrubato.h"
#include "essentiamath.h" // abs

using namespace std;
using namespace essentia;
using namespace standard;

const char* BpmRubato::name = "BpmRubato";
const char* BpmRubato::description = DOC("This algorithm extracts the locations of large tempo changes from a list of beat ticks.\n"
"\n"
"An exception is thrown if the input beats are not in ascending order and/or if the input beats contain duplicate values.\n"
"\n"
"Quality: experimental (non-reliable, poor accuracy).\n"
"\n"
"References:\n"
"  [1] Tempo Rubato - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Rubato");

void BpmRubato::configure() {
  _tolerance = parameter("tolerance").toReal();
  _longestRegion = parameter("longRegionsPruningTime").toReal();
  _shortestRegion = parameter("shortRegionsMergingTime").toReal();
}

void BpmRubato::compute() {
  const vector<Real>& beats = _beats.get();
  vector<Real>& rubatoStart = _rubatoStart.get();
  vector<Real>& rubatoStop  = _rubatoStop.get();
  int& rubatoNumber = _rubatoNumber.get();
  _rubatoOn = false; // true means currently in a region of changing rhythm

  int size = int(beats.size());
  rubatoStart.resize(0); rubatoStop.resize(0);

  if (size > 6) {

    int i = 5;
    if (beats[i]  <= beats[i-1]
      || beats[i-1] <= beats[i-2]
      || beats[i-2] <= beats[i-3]
      || beats[i-3] <= beats[i-4]
      || beats[i-4] <= beats[i-5]) {
      throw EssentiaException("BpmRubato: beat ticks must be in ascending order and must not contain duplicates");
    }

    // compute distances between ticks
    Real tmp1 = 60./ (beats[i  ] - beats[i-1]);
    Real tmp2 = 60./ (beats[i-1] - beats[i-2]);
    Real tmp3 = 60./ (beats[i-2] - beats[i-3]);
    Real tmp4 = 60./ (beats[i-3] - beats[i-4]);
    Real tmp5 = 60./ (beats[i-4] - beats[i-5]);

    for (i = 6; i < size; ++i) {
      /*    __   look for steps
       *      \__
       */
      if (abs(1. - tmp1 / tmp4) >= _tolerance    // not sure if this is a good way to measure change
          && abs(1. - tmp2 / tmp5) >= _tolerance
          && abs(1. - tmp2 / tmp4) >= _tolerance
          && abs(1. - tmp1 / tmp5) >= _tolerance
          && abs(1. - tmp1 / tmp2) <= _tolerance
          && abs(1. - tmp4 / tmp5) <= _tolerance) {

        // opening region
        if (!_rubatoOn) {
          // first region to open
          if (rubatoStop.empty()) {
            rubatoStart.push_back(beats[i-2]);
          }
          // do not open the region if closed one shortly before
          else if (beats[i-2] - rubatoStop.back() < _shortestRegion) {
            rubatoStop.pop_back();
          }
          // otherwise go ahead and open it
          else {
            rubatoStart.push_back(beats[i-2]);
          }
          // we have now entered a rubato region
          _rubatoOn = true;
        }
        // closing region
        else {
          if (!rubatoStart.empty()
              && beats[i-2] - rubatoStart.back() > _longestRegion) {
            // last start was too far, this was not a rubato region
            // not adding stop, removing last start
            // starting new region
            rubatoStart.pop_back();
            rubatoStart.push_back(beats[i-2]);
          }
          else if (!rubatoStop.empty() &&
                   beats[i-2] - rubatoStop.back() < _shortestRegion) {
            // last stop was too close, merging rubato regions
            // not adding stop, removing stop
            rubatoStop.pop_back();
          }
          else {
            // first stop
            rubatoStop.push_back(beats[i-2]);
            _rubatoOn = false;
          }
        }
      }

      // rotate and compute the next periods in the neighbourhood
      tmp5 = tmp4; tmp4 = tmp3; tmp3 = tmp2; tmp2 = tmp1;

      if (beats[i] <= beats[i-1]) {
        throw EssentiaException("BpmRubato: beat ticks must be in ascending order and must not contain duplicates");
      }

      tmp1 = 60./ (beats[i] - beats[i-1]);
    }

    // close the last region, if still in rubato region
    if (_rubatoOn) {
      if (!rubatoStart.empty() &&
          beats.back() - rubatoStart.back() > _longestRegion) {
        // last start was too far, this was not a rubato region
        // not adding stop, removing start
        rubatoStart.pop_back();
      }
      else {
        // closing open region with last beat
        rubatoStop.push_back(beats.back());
      }
      _rubatoOn = false;
    }
  }
  rubatoNumber = (int) rubatoStop.size();
}
