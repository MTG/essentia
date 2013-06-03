/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "stereodemuxer.h"
#include "sourcebase.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* StereoDemuxer::name = "StereoDemuxer";
const char* StereoDemuxer::description = DOC(
"Given a stereo signal, this algorithm outputs left and right channel separately."
"If the signal is monophonic, it outputs a zero signal on the right channel."
);

AlgorithmStatus StereoDemuxer::process() {
  EXEC_DEBUG("process()");
  AlgorithmStatus status = acquireData();
  EXEC_DEBUG("data acquired");

  if (status != OK) {
    // if shouldStop is true, that means there is no more audio, so we need
    // to take what's left to fill in half-frames, instead of waiting for more
    // data to come in (which would have done by returning from this function)
    if (!shouldStop()) return NO_INPUT;

    int available = input("audio").available();
    if (available == 0) return NO_INPUT;

    input("audio").setAcquireSize(available);
    input("audio").setReleaseSize(available);

    output("left").setAcquireSize(available);
    output("left").setReleaseSize(available);
    output("right").setAcquireSize(available);
    output("right").setReleaseSize(available);

    return process();
  }

  const vector<StereoSample>& audio = _audio.tokens();
  vector<AudioSample>& left = _left.tokens();
  vector<AudioSample>& right = _right.tokens();

  for (int i=0; i<(int)audio.size(); i++) {
    left[i] = audio[i].left();
    right[i] = audio[i].right();
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

const char* StereoDemuxer::name = "StereoDemuxer";
const char* StereoDemuxer::description = DOC(
"Given a stereo signal, this algorithm outputs left and right channel separately."
"If the signal is monophonic, it outputs a zero signal on the right channel."
);

void StereoDemuxer::createInnerNetwork() {
  _demuxer = streaming::AlgorithmFactory::create("StereoDemuxer");

  _audiogen = new streaming::VectorInput<StereoSample, 4096>();
  _leftStorage = new streaming::VectorOutput<AudioSample>();
  _rightStorage = new streaming::VectorOutput<AudioSample>();

  _audiogen->output("data")  >>  _demuxer->input("audio");
  _demuxer->output("left")   >>  _leftStorage->input("data");
  _demuxer->output("right")  >>  _rightStorage->input("data");

  _network = new scheduler::Network(_audiogen);
}

void StereoDemuxer::compute() {
  const vector<StereoSample>& audio = _audio.get();
  vector<AudioSample>& left = _left.get();
  vector<AudioSample>& right = _right.get();

  _audiogen->setVector(&audio);
  _leftStorage->setVector(&left);
  _rightStorage->setVector(&right);

  _network->run();
}

} // namespace standard
} // namespace essentia
