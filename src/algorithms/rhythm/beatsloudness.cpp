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

#include "beatsloudness.h"
#include "algorithmfactory.h"

using namespace std;


namespace essentia {
namespace streaming {

const char* BeatsLoudness::name = "BeatsLoudness";
const char* BeatsLoudness::description = DOC("Calculates the loudness computed only on the beats, both on the whole frequency range and on each specified frequency band. See the Loudness algorithm for a description of loudness and SingleBeatLoudness for a more detailed explanation.");

BeatsLoudness::BeatsLoudness() {
  // visible interface
  declareInput(_signal, "signal", "the input audio signal");

  declareOutput(_loudness, "loudness", "the beat's energy in the whole spectrum");
  declareOutput(_loudnessBandRatio, "loudnessBandRatio", "the ratio of the beat's energy in each band");

  // inner network
  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  _slicer = factory.create("Slicer");
  _beatLoud = factory.create("SingleBeatLoudness");

  _signal                                 >>  _slicer->input("audio");
  _slicer->output("frame")                >>  _beatLoud->input("beat");
  _beatLoud->output("loudness")           >>  _loudness;
  _beatLoud->output("loudnessBandRatio")  >>  _loudnessBandRatio;
}


void BeatsLoudness::configure() {
  // Slicing around the beats to be able to compute the beats loudness
  Real beatWindowDuration = parameter("beatWindowDuration").toReal();
  Real beatDuration = parameter("beatDuration").toReal();

  vector<Real> ticks = parameter("beats").toVectorReal();
  vector<Real> startTimes(ticks.size()), endTimes(ticks.size());

  for (int i=0; i<int(ticks.size()); ++i) {
    startTimes[i] = ticks[i] - beatWindowDuration/2.0;
    // make sure we don't cause an assert to fail because we missed one sample
    // due to rounding errors...
    endTimes[i] = ticks[i] + beatWindowDuration/2.0 + beatDuration + 0.0001;

    // in case the window started before the beginning of the sound, slide it
    // just what's needed to the right (temporally speaking)
    if (startTimes[i] < 0.0) {
      Real adjust = -startTimes[i];
      startTimes[i] += adjust;
      endTimes[i] += adjust;
    }
  }

  _slicer->configure(INHERIT("sampleRate"),
                     "startTimes", startTimes,
                     "endTimes", endTimes);

  _beatLoud->configure(INHERIT("sampleRate"),
                       INHERIT("beatWindowDuration"),
                       INHERIT("beatDuration"),
                       INHERIT("frequencyBands"));

}


} // namespace streaming
} // namespace essentia


#include "network.h"
#include "poolstorage.h"

namespace essentia {
namespace standard {

const char* BeatsLoudness::name = "BeatsLoudness";
const char* BeatsLoudness::description = DOC("Calculates the loudness computed only on the beats, both on the whole frequency range and only the bass frequencies. See the Loudness algorithm for a description of loudness.");

void BeatsLoudness::createInnerNetwork() {
  _beatLoud = streaming::AlgorithmFactory::create("BeatsLoudness");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _beatLoud->input("signal");
  _beatLoud->output("loudness")           >>  PC(_pool, "internal.loudness");
  _beatLoud->output("loudnessBandRatio")  >>  PC(_pool, "internal.loudnessBandRatio");

  _network = new scheduler::Network(_vectorInput);
}

void BeatsLoudness::configure() {
  _beatLoud->configure(INHERIT("sampleRate"),
                       INHERIT("beats"),
                       INHERIT("beatWindowDuration"),
                       INHERIT("beatDuration"),
                       INHERIT("frequencyBands"));
}

void BeatsLoudness::compute() {
  const vector<Real>& signal = _signal.get();
  if (signal.empty())
    throw EssentiaException("BeatsLoudness: Cannot compute loudness of an empty signal");

  vector<Real>& loudness = _loudness.get();
  vector<vector<Real> >& loudnessBand = _loudnessBand.get();

  _vectorInput->setVector(&signal);

  _network->run();
  try {
    loudness = _pool.value<vector<Real> >("internal.loudness");
    loudnessBand = _pool.value<vector<vector<Real> > >("internal.loudnessBandRatio");
  }
  catch (EssentiaException&) {
    // probably beats were not specified, or slicer did not produce any windows
    // due to mismatch between beat positions and the duration of audio
    loudness.clear();
    loudnessBand.clear();
  }
}

BeatsLoudness::~BeatsLoudness() {
  delete _network;
}

} // namespace standard
} // namespace essentia
