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

#include "tuningfrequencyextractor.h"
#include "algorithmfactory.h"
#include "poolstorage.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* TuningFrequencyExtractor::name = "TuningFrequencyExtractor";
const char* TuningFrequencyExtractor::description = DOC("This algorithm extracts the tuning frequency of an audio signal");

TuningFrequencyExtractor::TuningFrequencyExtractor(): _frameCutter(0), _spectralPeaks(0), _spectrum(0), _tuningFrequency(0), _windowing(0) {
  createInnerNetwork();
}

void TuningFrequencyExtractor::createInnerNetwork() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter     = factory.create("FrameCutter");
  _spectralPeaks   = factory.create("SpectralPeaks");
  _spectrum        = factory.create("Spectrum");
  _tuningFrequency = factory.create("TuningFrequency");
  _windowing       = factory.create("Windowing");

  _windowing->configure("type", "blackmanharris62");
  _spectralPeaks->configure("orderBy", "frequency",
                            "magnitudeThreshold", 1e-05,
                            "minFrequency", 40,
                            "maxFrequency", 5000,
                            "maxPeaks", 10000);

  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_tuningFreq, "tuningFrequency", "the computed tuning frequency");

  attach(_signal, _frameCutter->input("signal"));

  _frameCutter->output("frame")            >>  _windowing->input("frame");
  _windowing->output("frame")              >>  _spectrum->input("frame");
  _spectrum->output("spectrum")            >>  _spectralPeaks->input("spectrum");
  _spectralPeaks->output("frequencies")    >>  _tuningFrequency->input("frequencies");
  _spectralPeaks->output("magnitudes")     >>  _tuningFrequency->input("magnitudes");
  connect(_tuningFrequency->output("tuningCents"), NOWHERE);

  attach(_tuningFrequency->output("tuningFrequency"), _tuningFreq);
}

void TuningFrequencyExtractor::configure() {
  int frameSize = parameter("frameSize").toInt();
  int hopSize = parameter("hopSize").toInt();
  _frameCutter->configure("silentFrames", "noise", "hopSize", hopSize, "frameSize", frameSize);
}


TuningFrequencyExtractor::~TuningFrequencyExtractor() {
  delete _frameCutter;
  delete _spectralPeaks;
  delete _spectrum;
  delete _tuningFrequency;
  delete _windowing;
}

} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

const char* TuningFrequencyExtractor::name = "TuningFrequencyExtractor";
const char* TuningFrequencyExtractor::description = DOC("this algorithm extracts the tuning frequency of an audio signal");

TuningFrequencyExtractor::TuningFrequencyExtractor() {
  declareInput(_signal, "signal", "the audio input signal");
  declareOutput(_tuningFrequency, "tuningFrequency", "the computed tuning frequency");

  _tuningFrequencyExtractor = streaming::AlgorithmFactory::create("TuningFrequencyExtractor");
  _vectorInput = new streaming::VectorInput<Real>();
  createInnerNetwork();
}

TuningFrequencyExtractor::~TuningFrequencyExtractor() {
  delete _network;
}

void TuningFrequencyExtractor::reset() {
  _network->reset();
}

void TuningFrequencyExtractor::configure() {
  _tuningFrequencyExtractor->configure(INHERIT("frameSize"), INHERIT("hopSize"));
}

void TuningFrequencyExtractor::createInnerNetwork() {
  *_vectorInput  >>  _tuningFrequencyExtractor->input("signal");
  _tuningFrequencyExtractor->output("tuningFrequency")  >>  PC(_pool, "tuningFrequency");

  _network = new scheduler::Network(_vectorInput);
}

void TuningFrequencyExtractor::compute() {
  const vector<Real>& signal = _signal.get();
  _vectorInput->setVector(&signal);

  _network->run();

  vector<Real>& frequency = _tuningFrequency.get();

  frequency = _pool.value<vector<Real> >("tuningFrequency");
}

} // namespace standard
} // namespace essentia

