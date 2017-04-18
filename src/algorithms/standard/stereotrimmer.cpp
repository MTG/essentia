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

#include "stereotrimmer.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* StereoTrimmer::name = "StereoTrimmer";
const char* StereoTrimmer::category = "Standard";
const char* StereoTrimmer::description = DOC("This algorithm extracts a segment of a stereo audio signal given its start and end times.\n"
"Giving \"startTime\" greater than \"endTime\" will raise an exception.");

void StereoTrimmer::configure() {
  Real sampleRate = parameter("sampleRate").toReal();
  _startIndex = (long long)(parameter("startTime").toReal() * sampleRate);
  _endIndex = (long long)(parameter("endTime").toReal() * sampleRate);
  if (_startIndex > _endIndex) {
    throw EssentiaException("StereoTrimmer: startTime cannot be larger than endTime.");
  }
  _checkRange = parameter("checkRange").toBool();
}

void StereoTrimmer::compute() {
  const vector<StereoSample>& input = _input.get();
  vector<StereoSample>& output = _output.get();
  int size = input.size();

  if (_startIndex < 0) _startIndex = 0;
  if (_startIndex > size) {
    if (_checkRange) {
      throw EssentiaException("StereoTrimmer: cannot trim beyond the size of the input signal");
    }
    _startIndex = size;
    E_WARNING("StereoTrimmer: empty output due to insufficient input signal size");
  }
  if (_endIndex > size) _endIndex = size;

  size = _endIndex-_startIndex;
  output.resize(size);
  memcpy(&output[0], &input[0]+_startIndex, size*sizeof(StereoSample));
}

} // namespace essentia
} // namespace standard


namespace essentia {
namespace streaming {

const char* StereoTrimmer::name = essentia::standard::StereoTrimmer::name;
const char* StereoTrimmer::category = essentia::standard::StereoTrimmer::category;
const char* StereoTrimmer::description = essentia::standard::StereoTrimmer::description;

void StereoTrimmer::configure() {
  Real sampleRate = parameter("sampleRate").toReal();
  _startIndex = (long long)(parameter("startTime").toReal() * sampleRate);
  _endIndex = (long long)(parameter("endTime").toReal() * sampleRate);
  if (_startIndex > _endIndex) {
    throw EssentiaException("StereoTrimmer: startTime cannot be larger than endTime.");
  }
  _consumed = 0;
  _preferredSize = defaultPreferredSize;
}

AlgorithmStatus StereoTrimmer::process() {
  EXEC_DEBUG("process()");

  if ((_consumed < _startIndex) && (_consumed + _preferredSize > _startIndex)) {
    _input.setAcquireSize(_startIndex - _consumed);
    _input.setReleaseSize(_startIndex - _consumed);
  }

  if (_consumed == _startIndex) {
    _input.setAcquireSize(_preferredSize);
    _input.setReleaseSize(_preferredSize);
  }

  AlgorithmStatus status = acquireData();

  if (status != OK) {
    // if status == NO_OUTPUT, we should temporarily stop the framecutter,
    // return from this function so its dependencies can process the frames,
    // and reschedule the framecutter to run when all this is done.
    if (status == NO_OUTPUT) {
      EXEC_DEBUG("no more output available for trimmer; mark it for rescheduling and return");
      //_reschedule = true;
      return NO_OUTPUT; // if the buffer is full, we need to have produced something!
    }

    // if shouldStop is true, that means there is no more audio, so we need
    // to take what's left to fill in the output buffer
    if (!shouldStop()) return NO_INPUT;

    int available = input("signal").available();
    EXEC_DEBUG("Frame could not be fully acquired. Next frame will be incomplete");
    EXEC_DEBUG("There are " << available << " available tokens");
    if (available == 0) {
      shouldStop(true);
      return NO_INPUT;
    }

    _input.setAcquireSize(available);
    _input.setReleaseSize(available);
    _output.setAcquireSize(available);
    _output.setReleaseSize(available);
    _preferredSize = available;
    return process();
  }

  EXEC_DEBUG("data acquired");


  // get the audio input and copy it to the output
  const vector<StereoSample>& input = _input.tokens();
  vector<StereoSample>& output = _output.tokens();


  if (_consumed >= _startIndex && _consumed < _endIndex) {
    assert(input.size() == output.size());
    int howMany = min((long long)input.size(), _endIndex - _consumed);
    fastcopy(output.begin(), input.begin(), howMany);

    _output.setReleaseSize(howMany);
  }
  else {
    _output.setReleaseSize(0);
  }

  EXEC_DEBUG("produced frame");

  _consumed += _input.releaseSize();

  // optimization: we should also tell the parent (most of the time an
  // audio loader) to also stop, to avoid decoding an entire mp3 when only
  // 10 seconds are needed
  if (_consumed >= _endIndex) {
    // FIXME: does still still work with the new composites?
    shouldStop(true);
    const_cast<SourceBase*>(_input.source())->parent()->shouldStop(true);
  }


  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}


void StereoTrimmer::reset() {
  Algorithm::reset();
  _consumed = 0;
  _preferredSize = defaultPreferredSize;

  // make sure to reset I/O sizes
  _input.setAcquireSize(_preferredSize);
  _input.setReleaseSize(_preferredSize);
  _output.setAcquireSize(_preferredSize);
  _output.setReleaseSize(_preferredSize);
} 

} // namespace streaming
} // namespace essentia
