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

#include "monomixer.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* MonoMixer::name = "MonoMixer";
const char* MonoMixer::description = DOC("Given a stereo signal, this algorithm downmixes the signal into a single channel. If the signal was already a monoaural, it is left unchanged.\n"
"\n"
"References:\n"
"  [1] downmixing - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Downmixing\n");

void MonoMixer::compute() {
  const vector<StereoSample>& input = _inputAudio.get();
  const int& nChannels = _channels.get();
  vector<Real>& output = _outputAudio.get();

  int size = input.size();
  output.resize(size);
  if (nChannels == 1) {
    for (int i=0; i<size; i++) {
      output[i] = input[i].left();
    }
    return;
  }
  if (_type =="mix") {
    for (int i=0; i<size; ++i) {
      output[i] = 0.5*(input[i].left()+input[i].right());
    }
  }
  else if (_type =="left") {
    for (int i=0; i<size; ++i) {
      output[i] = input[i].left();
    }
  }
  else if (_type =="right") {
    for (int i=0; i<size; ++i) {
      output[i] = input[i].right();
    }
  }
  else //should never get here
    throw EssentiaException("MonoMixer: Uknown downmixing type");
}

} // namespace standard
} // namespace essentia

namespace essentia {
namespace streaming {

const char* MonoMixer::name = standard::MonoMixer::name;
const char* MonoMixer::description = standard::MonoMixer::description;

AlgorithmStatus MonoMixer::process() {

  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired (in: " << _inputAudio.acquireSize()
             << " - out: " << _outputAudio.acquireSize() << ")");

  if (status != OK) {
    if (!shouldStop()) return status;

    // if shouldStop is true, that means there is no more audio coming, so we need
    // to take what's left to fill in half-frames, instead of waiting for more
    // data to come in (which would have done by returning from this function)

    int available = input("audio").available();
    if (available == 0) return NO_INPUT;

    input("audio").setAcquireSize(available);
    input("audio").setReleaseSize(available);

    output("audio").setAcquireSize(available);
    output("audio").setReleaseSize(available);

    return process();
  }


  int nChannels = _channels.lastTokenProduced();
  const vector<StereoSample>& input = _inputAudio.tokens();
  vector<AudioSample>& output = _outputAudio.tokens();

  if (nChannels == 1) {
    for (int i=0; i<int(input.size()); i++) {
      output[i] = input[i].left();
    }
  }
  else {
    if (_type == "mix") {
      for (int i=0; i<int(input.size()); i++) {
        output[i] = (input[i].left() + input[i].right()) * 0.5;
      }
    }
    else if (_type == "left") {
      for (int i=0; i<int(input.size()); i++) {
        output[i] = input[i].left();
      }
    }
    else if (_type == "right") {
      for (int i=0; i<int(input.size()); i++) {
        output[i] = input[i].right();
      }
    }
    else {
      // should never be able to arrive here
      throw EssentiaException("MonoMixer: Uknown downmixing type");
    }
  }

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}

} // namespace streaming
} // namespace essentia
