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

#include "slicer.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* Slicer::name = "Slicer";
const char* Slicer::description = DOC("This algorithm returns a vector of slices, which start and end times are given as parameters.\n"
"\n"
"The parameters, \"startTimes\" and \"endTimes\" must be coherent. If these parameters differ in size, an exception is thrown. If a particular startTime is larger than its corresponding endTime, an exception is thrown.");

void Slicer::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _startTimes = parameter("startTimes").toVectorReal();
  _endTimes = parameter("endTimes").toVectorReal();
  _timeUnits = parameter("timeUnits").toString();

  if (_startTimes.size() != _endTimes.size()) {
    throw EssentiaException("Slicer: startTimes and endTimes do not have the same number of elements");
  }

  // check given times correspond to valid slices
  for (int i=0; i<int(_startTimes.size()); ++i) {
    if (_startTimes[i] > _endTimes[i]) {
      ostringstream msg;
      msg << "Slicer: Slice number " << i+1 << ": [" << _startTimes[i] << ", " << _endTimes[i] << "] is invalid because its start time is after its end time";
      throw EssentiaException(msg);
    }

    // if the time units are in seconds, we have to make sure that the
    // startTime[i]*sampleRate doesn't cause an overflow of int (31 bits)
    if (_timeUnits == "seconds" &&
        (double(_startTimes[i])*_sampleRate > 0xEFFFFFFF ||
         double(_endTimes[i])*_sampleRate > 0xEFFFFFFF)) {
      ostringstream msg;
      msg << "Slicer: start or end time, multiplied by the sampleRate (" <<
        _sampleRate << "Hz), is too large (greater than 31 bits): [" <<
        _startTimes[i] << "s, " << _endTimes[i] << "s]";
      throw EssentiaException(msg);
    }
  }

  _slices.clear();
  _slices.resize(_startTimes.size());

  if (_timeUnits == "samples") {
    for (int i=0; i<int(_slices.size()); ++i) {
      _slices[i] = make_pair(static_cast<int>(_startTimes[i]),
                             static_cast<int>(_endTimes[i]));
    }
  }
  else {
    for (int i=0; i<int(_slices.size()); ++i) {
      int s = int(_startTimes[i] * _sampleRate + 0.5);
      int e = s + int((_endTimes[i] - _startTimes[i]) * _sampleRate + 0.5);
      _slices[i] = make_pair(s, e);
    }
  }

  // set the acquireSize of the sink to the max size of the slices.
  // this will get overwritten as soon as we start processing, but is a hint
  // for automatic buffer resizing
  int maxSlice = defaultPreferredSize;
  for (int i=0; i<int(_slices.size()); ++i) {
    maxSlice = max(maxSlice, _slices[i].second - _slices[i].first);
  }

  _input.setAcquireSize(maxSlice);

  sort(_slices.begin(), _slices.end());

  reset();
}

void Slicer::reset() {
  Algorithm::reset();
  _consumed = 0;
  _sliceIdx = 0;
}


AlgorithmStatus Slicer::process() {
  EXEC_DEBUG("process()");

  // 10 first, consume and release tokens until we reach the start of the first slice
  // 20 produce the first slice, push it, and remove it from the list of slices to be processed
  // 30 goto 10

  // in case we already processed all slices, just gobble up data
  if (_sliceIdx == int(_slices.size())) {
    bool ok = _input.acquire(defaultPreferredSize);
    if (!ok) return NO_INPUT; // TODO: make a special case for end of stream?
    _input.release(defaultPreferredSize);
    return OK;
  }

  int startIndex = _slices[_sliceIdx].first;
  int endIndex = _slices[_sliceIdx].second;

  // arriving there, only consume the tokens left before we reach the beginning of the slice
  if ((_consumed < startIndex) && (_consumed + _input.acquireSize() > startIndex)) {
    _input.setAcquireSize(startIndex - _consumed);
    _input.setReleaseSize(startIndex - _consumed);
  }
  // we're at the beginning of a slice, grab it entirely at once
  if (_consumed == startIndex) {
    _input.setAcquireSize(endIndex - startIndex);
  }

  AlgorithmStatus status = acquireData();

  if (status != OK) {
    // FIXME: what does this return do here, without a comment explaining why we now skip everything after it???
    return status;

    // if shouldStop is true, that means there is no more audio, so we need to stop
    if (!shouldStop()) return status;

    EXEC_DEBUG("Slice could not be fully acquired. Creating a partial slice to "
               "the end of the token stream.");
    EXEC_DEBUG("There are " << _input.available() << " available tokens left.");

    // Since there is no more tokens to consume, the last slice will be
    // partial (not to the end of endIndex)
    if (_input.available() == 0) return NO_INPUT;

    _input.setAcquireSize(_input.available());
    _input.setReleaseSize(_input.available());
    status = acquireData();

    // If the last of the tokens could not be acquired, abort
    if (status != OK) return status;
  }

  int acquired = _input.acquireSize();

  EXEC_DEBUG("data acquired (in: " << acquired << ")");

  // are we dropping the tokens, or are we at the beginning of a slice, in which
  // case we need to copy it
  if (_consumed != startIndex) {
    // we're still dropping tokens to arrive to a slice
    _input.release(acquired);
    _consumed += acquired;

    return OK;
  }

  // we are copying a slice, get the audio input and copy it to the output
  const vector<Real>& input = _input.tokens();
  vector<Real>& output = _output.firstToken();

  assert((int)input.size() == _input.acquireSize());
  output.resize(input.size());

  fastcopy(output.begin(), input.begin(), (int)output.size());

  EXEC_DEBUG("produced frame");

  // set release size in function of next slice to get
  _sliceIdx++;

  int toRelease = acquired;

  // if next slice is very close, be careful not to release too many tokens
  if (_sliceIdx < (int)_slices.size()) {
    toRelease = min(toRelease, _slices[_sliceIdx].first - _consumed);
  }

  _input.setReleaseSize(toRelease);

  EXEC_DEBUG("releasing");
  releaseData();

  _consumed += _input.releaseSize();
  EXEC_DEBUG("released");

  // reset acquireSize to its default value
  _input.setAcquireSize(defaultPreferredSize);

  return OK;
}

} // namespace streaming
} // namespace essentia

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

const char* Slicer::name = "Slicer";
const char* Slicer::description = DOC("This algorithm returns a vector of slices, which start and end times are given as parameters.\n"
"\n"
"The parameters, \"startTimes\" and \"endTimes\" must be coherent. If these parameters differ in size, an exception is thrown. If a particular startTime is larger than its corresponding endTime, an exception is thrown.");

void Slicer::configure() {
  _slicer->configure(INHERIT("sampleRate"),
                     INHERIT("startTimes"),
                     INHERIT("endTimes"),
                     INHERIT("timeUnits"));
}

void Slicer::createInnerNetwork() {
  _slicer = streaming::AlgorithmFactory::create("Slicer");
  _storage = new streaming::VectorOutput<vector<Real> >();
  _gen = new streaming::VectorInput<Real>();

  *_gen                     >>  _slicer->input("audio");
  _slicer->output("frame")  >>  *_storage;

  _network = new scheduler::Network(_gen);
}

void Slicer::compute() {
  const vector<Real>& audio = _audio.get();
  vector<vector<Real> >& output = _output.get();
  output.clear();

  _gen->setVector(&audio);
  _storage->setVector(&output);

  _network->run();
}

} // namespace standard
} // namespace essentia
