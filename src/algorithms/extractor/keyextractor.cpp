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

#include "keyextractor.h"
#include "algorithmfactory.h"
#include "poolstorage.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* KeyExtractor::name = essentia::standard::KeyExtractor::name;
const char* KeyExtractor::category = essentia::standard::KeyExtractor::category;
const char* KeyExtractor::description = essentia::standard::KeyExtractor::description;


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
  _frameCutter       = factory.create("FrameCutter");
  _windowing         = factory.create("Windowing");
  _spectrum          = factory.create("Spectrum");
  _spectralPeaks     = factory.create("SpectralPeaks");
  _spectralWhitening = factory.create("SpectralWhitening");
  _hpcpKey           = factory.create("HPCP");
  _key               = factory.create("Key");

  // attach input proxy(ies)
  _audio  >>  _frameCutter->input("signal");

  // connect inner algorithms
  _frameCutter->output("frame")              >>  _windowing->input("frame");
  _windowing->output("frame")                >>  _spectrum->input("frame");
  _spectrum->output("spectrum")              >>  _spectralPeaks->input("spectrum");
  _spectrum->output("spectrum")              >>  _spectralWhitening->input("spectrum");
  _spectralPeaks->output("magnitudes")       >>  _spectralWhitening->input("magnitudes");
  _spectralPeaks->output("frequencies")      >>  _spectralWhitening->input("frequencies");
  _spectralWhitening->output("magnitudes")   >>  _hpcpKey->input("magnitudes");
  _spectralPeaks->output("frequencies")      >>  _hpcpKey->input("frequencies");
  _hpcpKey->output("hpcp")                   >>  _key->input("pcp");

  // attach output proxy(ies)
  _key->output("key")       >>  _keyKey;
  _key->output("scale")     >>  _keyScale;
  _key->output("strength")  >>  _keyStrength;

  _network = new scheduler::Network(_frameCutter);
}

void KeyExtractor::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _windowType = parameter("windowType").toString();
  _minFrequency = parameter("minFrequency").toReal();
  _maxFrequency = parameter("maxFrequency").toReal();
  _spectralPeaksThreshold = parameter("spectralPeaksThreshold").toReal();
  _maxPeaks = parameter("maximumSpectralPeaks").toReal();
  _hpcpSize = parameter("hpcpSize").toInt();
  _weightType = parameter("weightType").toString();
  _tuningFrequency = parameter("tuningFrequency").toReal();
  _pcpThreshold = parameter("pcpThreshold").toReal();
  _averageDetuningCorrection = parameter("averageDetuningCorrection").toBool();
  _profileType = parameter("profileType").toString();


  _frameCutter->configure("frameSize", _frameSize,
                          "hopSize", _hopSize);

  _windowing->configure("size", _frameSize,
                        "type", _windowType);

  _spectralPeaks->configure("orderBy", "magnitude",
                            "magnitudeThreshold", _spectralPeaksThreshold,
                            "minFrequency", _minFrequency,
                            "maxFrequency", _maxFrequency,
                            "maxPeaks", _maxPeaks,
                            "sampleRate", _sampleRate);

  _spectralWhitening->configure("maxFrequency", _maxFrequency,
                                "sampleRate", _sampleRate);

  _hpcpKey->configure("bandPreset", false,
                      "harmonics", 4,
                      "maxFrequency", _maxFrequency,
                      "minFrequency", _minFrequency, 
                      "nonLinear", false,
                      "normalized", "none",
                      "referenceFrequency", _tuningFrequency,
                      "sampleRate", _sampleRate,
                      "size", _hpcpSize, 
                      "weightType", _weightType,
                      "windowSize", 1.0,
                      "maxShifted", false);

  _key->configure("usePolyphony", false,
                  "useThreeChords", false,
                  "numHarmonics", 4, 
                  "slope",  0.6,
                  "profileType", _profileType,
                  "pcpSize", _hpcpSize,
                  "pcpThreshold", _pcpThreshold, 
                  "averageDetuningCorrection", _averageDetuningCorrection);

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
const char* KeyExtractor::category = "Tonal";
const char* KeyExtractor::description = DOC("This algorithm extracts key/scale for an audio signal. It computes HPCP frames for the input signal and applies key estimation using the Key algorithm.\n"
"\n"
"The algorithm allows tuning correction using two complementary methods:\n"
"  - Specify the expected `tuningFrequency` for the HPCP computation. The algorithm will adapt the semitone crossover frequencies for computing the HPCPs accordingly. If not specified, the default tuning is used. Tuning frequency can be estimated in advance using TuningFrequency algorithm.\n"
"  - Apply tuning correction posterior to HPCP computation, based on peaks in the HPCP distribution (`averageDetuningCorrection`). This is possible when hpcpSize > 12.\n"
"\n"
"For more information, see the HPCP and Key algorithms.");


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
  _keyExtractor->configure(INHERIT("sampleRate"),
                           INHERIT("frameSize"),
                           INHERIT("hopSize"),
                           INHERIT("windowType"),
                           INHERIT("minFrequency"),
                           INHERIT("maxFrequency"),
                           INHERIT("spectralPeaksThreshold"),
                           INHERIT("maximumSpectralPeaks"),
                           INHERIT("hpcpSize"),
                           INHERIT("weightType"),
                           INHERIT("tuningFrequency"),
                           INHERIT("pcpThreshold"),
                           INHERIT("averageDetuningCorrection"),
                           INHERIT("profileType"));
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
