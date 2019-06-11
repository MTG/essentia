/*
 * Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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

#include "falsestereodetector.h"

using namespace std;

namespace essentia{
namespace standard{  

const char *FalseStereoDetector::name = "FalseStereoDetector";
const char *FalseStereoDetector::category = "Audio Problems";
const char *FalseStereoDetector::description = DOC(
  "This algorithm detects if a stereo track has duplicated channels (false stereo)."
  "It is based on the Pearson linear correlation coefficient and thus it is robust "
  "scaling and shifting between channels.");


void FalseStereoDetector::configure() {
  _silenceThreshold = db2pow(parameter("silenceThreshold").toReal());
  _correlationThreshold = parameter("correlationThreshold").toReal();
}


void FalseStereoDetector::compute() {
  const vector<StereoSample> frame = _frame.get();
  int &isFalseStereo = _isFalseStereo.get();
  Real &correlation = _correlation.get();

  isFalseStereo = 0;
  correlation = 0.f;

  vector<Real> left;
  vector<Real> right;

  _demuxer->input("audio").set(frame);
  _demuxer->output("left").set(left);
  _demuxer->output("right").set(right);
  _demuxer->compute();
  _demuxer->reset();


  // if both channels are silent we can not state that there is 
  // no false stereo as the differece can be originated by the 
  // dithering noise and not because of the signal
  if ((instantPower(left) < _silenceThreshold) && 
      (instantPower(right) < _silenceThreshold)) {
    return;
  }

  correlation = pearsonCorrelationCoefficient(left, right);

  if (correlation > _correlationThreshold) {
    isFalseStereo = 1;
  }
}

} // namespace standard 
} // namespace essentia


#include "poolstorage.h"
#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

const char* FalseStereoDetector::name = standard::FalseStereoDetector::name;
const char* FalseStereoDetector::category = standard::FalseStereoDetector::category;
const char* FalseStereoDetector::description = standard::FalseStereoDetector::description;


FalseStereoDetector::FalseStereoDetector() : Algorithm() {

  _FalseStereoDetectorAlgo = standard::AlgorithmFactory::create("FalseStereoDetector");

  declareInput(_audio, "audio", "the input audio");
  declareOutput(_isFalseStereo, "isFalseStereo", "a flag indicating if the frame channes are simmilar");
  declareOutput(_correlation, "correlation", "correlation betweeen the input channels");
}


FalseStereoDetector::~FalseStereoDetector() {
  delete _FalseStereoDetectorAlgo;
}


AlgorithmStatus FalseStereoDetector::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired (in: " << _audio.acquireSize()
             << " - out: " << _audio.acquireSize() << ")");

  if (status != OK) {
    if (!shouldStop()) return status;

    // If shouldStop is true, that means there is no more audio coming, so we need
    // to take what's left to fill in half-frames, instead of waiting for more
    // data to come in (which would have done by returning from this function).
    int available = input("audio").available();
    if (available == 0) return NO_INPUT;

    input("audio").setAcquireSize(available);
    input("audio").setReleaseSize(available);

    return process();
  }

  const vector<StereoSample>& audio = _audio.tokens();
  vector<int>& isFalseStereo = _isFalseStereo.tokens();
  vector<Real>& correlation = _correlation.tokens();

  _FalseStereoDetectorAlgo->input("frame").set(audio);
  _FalseStereoDetectorAlgo->output("isFalseStereo").set(isFalseStereo[0]);
  _FalseStereoDetectorAlgo->output("correlation").set(correlation[0]);
  _FalseStereoDetectorAlgo->compute();

  EXEC_DEBUG("releasing");
  releaseData();
  EXEC_DEBUG("released");

  return OK;
}


void FalseStereoDetector::configure() {
  _frameSize = parameter("frameSize").toInt();

  _FalseStereoDetectorAlgo->configure(INHERIT("silenceThreshold"),
                                      INHERIT("correlationThreshold"));

  _audio.setAcquireSize(_frameSize);
  _audio.setReleaseSize(_frameSize);
  _isFalseStereo.setAcquireSize(1);
  _isFalseStereo.setReleaseSize(1);
  _correlation.setAcquireSize(1);
  _correlation.setReleaseSize(1);
}


void FalseStereoDetector::reset() {
  _audio.setAcquireSize(_frameSize);
  _audio.setReleaseSize(_frameSize);
  _isFalseStereo.setAcquireSize(1);
  _isFalseStereo.setReleaseSize(1);
  _correlation.setAcquireSize(1);
  _correlation.setReleaseSize(1);

  _FalseStereoDetectorAlgo->reset();
}

} // namespace streaming
} // namespace essentia
