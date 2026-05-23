/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#include "stereoresample.h"

using namespace std;

namespace essentia {
namespace standard {

const char* StereoResample::name = "StereoResample";
const char* StereoResample::category = "Standard";
const char* StereoResample::description = DOC("This algorithm resamples the input stereo signal to the desired sampling rate.\n\n"
"The quality of conversion is documented in [3].\n\n"
"This algorithm is only supported if essentia has been compiled with Real=float, otherwise it will throw an exception. It may also throw an exception if there is an internal error in the SRC library during conversion.\n\n"

"References:\n"
"  [1] Secret Rabbit Code, http://www.mega-nerd.com/SRC\n\n"
"  [2] Resampling - Wikipedia, the free encyclopedia\n"
"  http://en.wikipedia.org/wiki/Resampling\n\n"
"  [3] http://www.mega-nerd.com/SRC/api_misc.html#Converters");

void StereoResample::configure() {
  //_quality = parameter("quality").toInt();

  // create and configure algorithms
  _stereoDemuxer = AlgorithmFactory::create("StereoDemuxer");
  _stereoMuxer = AlgorithmFactory::create("StereoMuxer");
  _resample = AlgorithmFactory::create("Resample");

  _resample->configure(INHERIT("inputSampleRate"),
                       INHERIT("outputSampleRate"),
                       INHERIT("quality"));
}

void StereoResample::compute() {
  const std::vector<StereoSample>& signal = _signal.get();
  std::vector<StereoSample>& resampled = _resampled.get();

  // compute resampling for left and right channel
  _stereoDemuxer->input("audio").set(signal);
  _stereoDemuxer->output("left").set(_leftStorage);
  _stereoDemuxer->output("right").set(_rightStorage);

  _resample->input("signal").set(_leftStorage);
  _resample->output("signal").set(_left);

  _stereoDemuxer->compute();
  _resample->compute();

  _resample->input("signal").set(_rightStorage);
  _resample->output("signal").set(_right);

  _stereoMuxer->input("left").set(_left);
  _stereoMuxer->input("right").set(_right);
  _stereoMuxer->output("audio").set(resampled);

  _resample->compute();
  _stereoMuxer->compute();
}

} // namespace standard
} // namespace essentia


/*
namespace essentia {
namespace streaming {

const char* StereoResample::name = standard::StereoResample::name;
const char* StereoResample::description = standard::StereoResample::description;

StereoResample::StereoResample() : _configured(false) {
  _preferredSize = 4096; // arbitrary
  declareInput(_signal, _preferredSize, "signal", "the input stereo signal");
  declareOutput(_resampled, _preferredSize, "signal", "the stereo resampled signal");

  createInnerNetwork();
}

void StereoResample::createInnerNetwork(){
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _stereoDemuxer = AlgorithmFactory::create("StereoDemuxer");
  _stereoMuxer = AlgorithmFactory::create("StereoMuxer");
  _resample = AlgorithmFactory::create("Resample");

  _resample->configure(INHERIT("inputSampleRate"),
                       INHERIT("outputSampleRate"),
                       INHERIT("quality"));

  // wire algorithms
  _signal >> _stereoDemuxer->input("audio");
  _stereoDemuxer->output("left") >> _resample->input("signal");
  _resample->output("signal") >> _stereoMuxer->input("left");
  _stereoDemuxer->output("right") >> _resample->input("signal");
  _resample->output("signal") >> _stereoMuxer->input("right");
  _stereoMuxer->output("audio") >> _resampled;

  // create network
  _network = new scheduler::Network(_stereoDemuxer);
}

void StereoResample::configure() {
  // TODO: initialize algorithms
  // TODO: create network
  // create and configure algorithms
  _stereoDemuxer = AlgorithmFactory::create("StereoDemuxer");
  _stereoMuxer = AlgorithmFactory::create("StereoMuxer");
  _resample = AlgorithmFactory::create("Resample");

  _resample->configure(INHERIT("inputSampleRate"),
                       INHERIT("outputSampleRate"),
                       INHERIT("quality"));
}

void StereoResample::reset() {
  clearAlgos();
}

void StereoResample::clearAlgos() {
  if (!_configured) return;
  delete _network;
}

} // namespace streaming
} // namespace essentia
*/