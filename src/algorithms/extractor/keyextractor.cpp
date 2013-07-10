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

#include "keyextractor.h"
#include "algorithmfactory.h"
#include "poolstorage.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* KeyExtractor::name = "KeyExtractor";
const char* KeyExtractor::description = DOC("this algorithm extracts key/scale for an audio stream");

KeyExtractor::KeyExtractor(): _frameCutter(0), _windowing(0), _spectrum(0), _spectralPeaks(0),
                              _hpcpKey(0), _key(0) {

  declareInput(_audio, "audio", "the audio signal");
  declareOutput(_keyKey, "key", "see Key algorithm documentation");
  declareOutput(_keyScale, "scale", "see Key algorithm documentation");
  declareOutput(_keyStrength, "strength", "see Key algorithm documentation");

  createInnerNetwork();
}

void KeyExtractor::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  // instantiate all required algorithms
  _frameCutter   = factory.create("FrameCutter");
  _windowing     = factory.create("Windowing", "type", "blackmanharris62");
  _spectrum      = factory.create("Spectrum");
  _spectralPeaks = factory.create("SpectralPeaks",
                                  "orderBy", "magnitude", "magnitudeThreshold", 1e-05,
                                  "minFrequency", 40, "maxFrequency", 5000, "maxPeaks", 10000);
  _hpcpKey       = factory.create("HPCP");
  _key           = factory.create("Key");

  // attach input proxy(ies)
  _audio  >>  _frameCutter->input("signal");

  // connect inner algorithms
  _frameCutter->output("frame")          >>  _windowing->input("frame");
  _windowing->output("frame")            >>  _spectrum->input("frame");
  _spectrum->output("spectrum")          >>  _spectralPeaks->input("spectrum");
  _spectralPeaks->output("magnitudes")   >>  _hpcpKey->input("magnitudes");
  _spectralPeaks->output("frequencies")  >>  _hpcpKey->input("frequencies");
  _hpcpKey->output("hpcp")               >>  _key->input("pcp");

  // attach output proxy(ies)
  _key->output("key")       >>  _keyKey;
  _key->output("scale")     >>  _keyScale;
  _key->output("strength")  >>  _keyStrength;

  _network = new scheduler::Network(_frameCutter);
}

void KeyExtractor::configure() {
  int frameSize = parameter("frameSize").toInt();
  int hopSize = parameter("hopSize").toInt();
  Real tuningFrequency = parameter("tuningFrequency").toReal();

  _frameCutter->configure("frameSize", frameSize,
                          "hopSize", hopSize,
                          "silentFrames", "noise");

  _hpcpKey->configure("referenceFrequency", tuningFrequency,
                      "minFrequency", 40.0,
                      "nonLinear", false,
                      "maxFrequency", 5000.0,
                      "bandPreset", false,
                      "windowSize", 1.33333333333,
                      "weightType", "squaredCosine",
                      "size", 36);

  _configured = true;
}


KeyExtractor::~KeyExtractor() {
  delete _network;
}


} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

const char* KeyExtractor::name = "KeyExtractor";
const char* KeyExtractor::description = DOC("this algorithm extracts tonal features");

KeyExtractor::KeyExtractor() {
  declareInput(_audio, "audio", "the audio input signal");

  declareOutput(_key, "key", "See Key algorithm documentation");
  declareOutput(_scale, "scale", "See Key algorithm documentation");
  declareOutput(_strength, "strength", "See Key algorithm documentation");

  createInnerNetwork();
}

KeyExtractor::~KeyExtractor() {
  delete _network;
}

void KeyExtractor::reset() {
  _network->reset();
}

void KeyExtractor::configure() {
  _keyExtractor->configure(INHERIT("frameSize"),
                           INHERIT("hopSize"),
                           INHERIT("tuningFrequency"));
}

void KeyExtractor::createInnerNetwork() {
  _keyExtractor = streaming::AlgorithmFactory::create("KeyExtractor");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput                      >>  _keyExtractor->input("audio");

  _keyExtractor->output("key")       >>  PC(_pool, "key");
  _keyExtractor->output("scale")     >>  PC(_pool, "scale");
  _keyExtractor->output("strength")  >>  PC(_pool, "strength");

  _network = new scheduler::Network(_vectorInput);
}

void KeyExtractor::compute() {
  const vector<Real>& audio = _audio.get();
  _vectorInput->setVector(&audio);

  _network->run();

  string& key      = _key.get();
  string& scale    = _scale.get();
  Real&   strength = _strength.get();

  key      = _pool.value<string>("key");
  scale    = _pool.value<string>("scale");
  strength = _pool.value<Real>("strength");
}

} // namespace standard
} // namespace essentia
