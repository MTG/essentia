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

#include "startstopsilence.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* StartStopSilence::name = "StartStopSilence";
const char* StartStopSilence::description = DOC("This algorithm outputs the frame at which sound begins and the frame at which sound ends.");

void StartStopSilence::configure() {
    _threshold = db2pow(parameter("threshold").toReal());
};

void StartStopSilence::compute() {
  const vector<Real>& frame = _frame.get();
  int& start = _startSilenceSource.get();
  int& stop = _stopSilenceSource.get();

  // should output the first non-silent frame, thus:
  if (_wasSilent) {
    _startSilence++;
    _wasSilent = false;
  }

  bool silentFrame = instantPower(frame) < _threshold;
  if (silentFrame && (_stopSilence == _startSilence)) {
    _startSilence = _nFrame;
    _wasSilent = true;
  }

  if (!silentFrame) {
    _stopSilence = _nFrame;
  }


  if (_startSilence > _stopSilence) {
    _stopSilence = _startSilence;
  }

  start = _startSilence;
  stop = _stopSilence;
  _nFrame++;
}

void StartStopSilence::reset() {
  _startSilence = 0;
  _stopSilence = 0;
  _nFrame = 0;
  _wasSilent = false;
}

} // essentia standard
} // essentia namespace



namespace essentia {
namespace streaming {

const char* StartStopSilence::name = "StartStopSilence";
const char* StartStopSilence::description = DOC("This algorithm outputs the frame at which sound begins and the frame at which sound ends.");

void StartStopSilence::configure() {
  _startSilence = 0;
  _stopSilence = 0;
  _nFrame = 0;
  _threshold = db2pow(parameter("threshold").toReal());
}

AlgorithmStatus StartStopSilence::process() {
  EXEC_DEBUG("process()");

  AlgorithmStatus status = acquireData();

  if (status != OK) {
    if (!shouldStop()) return status;

    // should output the first non-silent frame, thus:
    if (_startSilence < _nFrame &&
        _startSilence != 0 &&
        _startSilence < _stopSilence) { _startSilence++; }

    // in case we have complete silence:
    if (_startSilence > _stopSilence) _stopSilence = _startSilence;

    _startSilenceSource.push(_startSilence);
    _stopSilenceSource.push(_stopSilence);

    return FINISHED;
  }

  const vector<Real>& frame = *(vector<Real>*)_frame.getFirstToken();
  bool silentFrame = instantPower(frame) < _threshold;

  if (silentFrame && (_stopSilence == 0)) {
    _startSilence = _nFrame;
  }

  if (!silentFrame) _stopSilence = _nFrame;

  releaseData();

  _nFrame++;

  return OK;
}

} // namespace streaming
} // namespace essentia
