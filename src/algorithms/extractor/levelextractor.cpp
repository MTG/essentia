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

#include "levelextractor.h"
#include "algorithmfactory.h"
#include "poolstorage.h"

using namespace std;

namespace essentia {
namespace streaming {


const char* LevelExtractor::name = "LevelExtractor";
const char* LevelExtractor::description = DOC("this algorithm extracts the loudness of an audio signal");

LevelExtractor::LevelExtractor() {

  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_loudnessValue, "loudness", "the loudness values");

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter = factory.create("FrameCutter",
                                "silentFrames", "noise",
                                "startFromZero", true);

  _loudness = factory.create("Loudness");

  _signal                        >>  _frameCutter->input("signal");
  _frameCutter->output("frame")  >>  _loudness->input("signal");
  _loudness->output("loudness")  >>  _loudnessValue;
}

void LevelExtractor::configure() {
  _frameCutter->configure(INHERIT("frameSize"),
                          INHERIT("hopSize"));
}

LevelExtractor::~LevelExtractor() {
  delete _frameCutter;
  delete _loudness;
}


} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

const char* LevelExtractor::name = "LevelExtractor";
const char* LevelExtractor::description = DOC("this algorithm extracts the loudness of an audio signal");

LevelExtractor::LevelExtractor() {
  declareInput(_signal, "signal", "the audio input signal");
  declareOutput(_loudness, "loudness", "the loudness values");

  createInnerNetwork();
}

LevelExtractor::~LevelExtractor() {
  delete _network;
}

void LevelExtractor::reset() {
  _network->reset();
  _pool.clear();
}

void LevelExtractor::configure() {
  _levelExtractor->configure(INHERIT("frameSize"), INHERIT("hopSize"));
}

void LevelExtractor::createInnerNetwork() {
  _levelExtractor = streaming::AlgorithmFactory::create("LevelExtractor");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput                        >>  _levelExtractor->input("signal");
  _levelExtractor->output("loudness")  >>  PC(_pool, "internal.loudness");

  _network = new scheduler::Network(_vectorInput);
}


 void LevelExtractor::compute() {
  const vector<Real>& signal = _signal.get();
  _vectorInput->setVector(&signal);

  _network->run();

  vector<Real>& loudness = _loudness.get();

  loudness = _pool.value<vector<Real> >("internal.loudness");
}

} // namespace standard
} // namespace essentia

