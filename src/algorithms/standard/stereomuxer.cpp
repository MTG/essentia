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

#include "stereomuxer.h"
#include "sourcebase.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* StereoMuxer::name = essentia::standard::StereoMuxer::name;
const char* StereoMuxer::category = essentia::standard::StereoMuxer::category;
const char* StereoMuxer::description = essentia::standard::StereoMuxer::description;


AlgorithmStatus StereoMuxer::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired");

  if (status != OK) {
    // if shouldStop is true, that means there is no more audio, so we need
    // to take what's left to fill in half-frames, instead of waiting for more
    // data to come in (which would have done by returning from this function)
    if (!shouldStop()) return NO_INPUT;

    int available = input("left").available();
    if (available == 0) return NO_INPUT;
    // TODO: make sure available is equal in left and right

    input("left").setAcquireSize(available);
    input("left").setReleaseSize(available);
    input("right").setAcquireSize(available);
    input("right").setReleaseSize(available);

    output("audio").setAcquireSize(available);
    output("audio").setReleaseSize(available);

    return process();
  }

  const vector<AudioSample>& left = _left.tokens();
  const vector<AudioSample>& right = _right.tokens();
  vector<StereoSample>& audio = _audio.tokens();

  for (int i=0; i<(int)left.size(); i++) {
    // TODO make sure left.size() always match right.size()
    audio[i].first = left[i];
    audio[i].second = right[i];
  }

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}

} // namespace streaming
} // namespace essentia

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

const char* StereoMuxer::name = "StereoMuxer";
const char* StereoMuxer::category = "Standard";
const char* StereoMuxer::description = DOC("This algorithm outputs a stereo signal given left and right channel separately.");


void StereoMuxer::compute() {
  const vector<AudioSample>& left = _left.get();
  const vector<AudioSample>& right = _right.get();
  vector<StereoSample>& audio = _audio.get();

  if (left.size() != right.size()) {
    throw EssentiaException("StereoMuxer: \"left\" and \"right\" inputs should contain equal number of audiosamples");
  }

  audio.resize(left.size());
  for (size_t i=0; i<left.size(); ++i) {
    audio[i].first = left[i];
    audio[i].second = right[i];
  }
}

} // namespace standard
} // namespace essentia
