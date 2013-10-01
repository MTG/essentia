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

#include "tonalextractor.h"
#include "algorithmfactory.h"
#include "poolstorage.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* TonalExtractor::name = "TonalExtractor";
const char* TonalExtractor::description = DOC("This algorithm extracts tonal features");

TonalExtractor::TonalExtractor(): _frameCutter(0), _windowing(0), _spectrum(0), _spectralPeaks(0),
                                  _hpcpKey(0), _hpcpChord(0), _hpcpTuning(0), _key(0),
                                  _chordsDescriptors(0), _chordsDetection(0) {

  declareInput(_signal, "signal", "the input audio signal");

  declareOutput(_chordsChangesRate, "chords_changes_rate", "See ChordsDescriptors algorithm documentation");
  declareOutput(_chordsHistogram, "chords_histogram", "See ChordsDescriptors algorithm documentation");
  declareOutput(_chordsKey, "chords_key", "See ChordsDescriptors algorithm documentation");
  declareOutput(_chordsNumberRate, "chords_number_rate", "See ChordsDescriptors algorithm documentation");
  declareOutput(_chordsProgression, "chords_progression", "See ChordsDetection algorithm documentation");
  declareOutput(_chordsScale, "chords_scale", "See ChordsDetection algorithm documentation");
  declareOutput(_chordsStrength, "chords_strength", "See ChordsDetection algorithm documentation");
  declareOutput(_hpcps, "hpcp", "See HPCP algorithm documentation");
  declareOutput(_hpcpsTuning, "hpcp_highres", "See HPCP algorithm documentation");
  declareOutput(_keyKey, "key_key", "See Key algorithm documentation");
  declareOutput(_keyScale, "key_scale", "See Key algorithm documentation");
  declareOutput(_keyStrength, "key_strength", "See Key algorithm documentation");

  createInnerNetwork();
}

void TonalExtractor::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _chordsDescriptors = factory.create("ChordsDescriptors");
  _chordsDetection   = factory.create("ChordsDetection");
  _key               = factory.create("Key");
  _spectralPeaks     = factory.create("SpectralPeaks",
                                      "orderBy", "magnitude", "magnitudeThreshold", 1e-05,
                                      "minFrequency", 40, "maxFrequency", 5000, "maxPeaks", 10000);

  _frameCutter       = factory.create("FrameCutter");
  _spectrum          = factory.create("Spectrum");
  _windowing         = factory.create("Windowing", "type", "blackmanharris62");
  _hpcpKey           = factory.create("HPCP");
  _hpcpChord         = factory.create("HPCP");
  _hpcpTuning        = factory.create("HPCP");

  _signal                                >>  _frameCutter->input("signal");

  _frameCutter->output("frame")          >>  _windowing->input("frame");
  _hpcpKey->output("hpcp")               >>  _key->input("pcp");
  _hpcpChord->output("hpcp")             >>  _chordsDetection->input("pcp");
  _chordsDetection->output("chords")     >>  _chordsDescriptors->input("chords");
  _key->output("key")                    >>  _chordsDescriptors->input("key");
  _key->output("scale")                  >>  _chordsDescriptors->input("scale");
  _spectralPeaks->output("magnitudes")   >>  _hpcpKey->input("magnitudes");
  _spectralPeaks->output("magnitudes")   >>  _hpcpChord->input("magnitudes");
  _spectralPeaks->output("magnitudes")   >>  _hpcpTuning->input("magnitudes");
  _spectralPeaks->output("frequencies")  >>  _hpcpKey->input("frequencies");
  _spectralPeaks->output("frequencies")  >>  _hpcpChord->input("frequencies");
  _spectralPeaks->output("frequencies")  >>  _hpcpTuning->input("frequencies");
  _spectrum->output("spectrum")          >>  _spectralPeaks->input("spectrum");
  _windowing->output("frame")            >>  _spectrum->input("frame");

  _chordsDescriptors->output("chordsChangesRate")  >>  _chordsChangesRate;
  _chordsDescriptors->output("chordsHistogram")    >>  _chordsHistogram;
  _chordsDescriptors->output("chordsKey")          >>  _chordsKey;
  _chordsDescriptors->output("chordsNumberRate")   >>  _chordsNumberRate;
  _chordsDetection->output("chords")               >>  _chordsProgression ;
  _chordsDescriptors->output("chordsScale")        >>  _chordsScale;
  _chordsDetection->output("strength")             >>  _chordsStrength;
  _hpcpKey->output("hpcp")                         >>  _hpcps;
  _hpcpTuning->output("hpcp")                      >>  _hpcpsTuning;
  _key->output("key")                              >>  _keyKey ;
  _key->output("scale")                            >>  _keyScale;
  _key->output("strength")                         >>  _keyStrength;

  _network = new scheduler::Network(_frameCutter);
}

void TonalExtractor::configure() {
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

  _hpcpChord->configure("referenceFrequency", tuningFrequency,
                        "minFrequency", 40.0,
                        "nonLinear", true,
                        "splitFrequency", 500.0,
                        "maxFrequency", 5000.0,
                        "bandPreset", true,
                        "windowSize", 0.5,
                        "weightType", "cosine",
                        "harmonics", 8,
                        "size", 36);

  _hpcpTuning->configure("referenceFrequency", tuningFrequency,
                         "minFrequency", 40.0,
                         "nonLinear", true,
                         "splitFrequency", 500.0,
                         "maxFrequency", 5000.0,
                         "bandPreset", true,
                         "windowSize", 0.5,
                         "weightType", "cosine",
                         "harmonics", 8,
                         "size", 120);
}


TonalExtractor::~TonalExtractor() {
  delete _network;
}


} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

const char* TonalExtractor::name = "TonalExtractor";
const char* TonalExtractor::description = DOC("this algorithm extracts tonal features");

TonalExtractor::TonalExtractor() {
  declareInput(_signal, "signal", "the audio input signal");

  declareOutput(_chordsChangesRate, "chords_changes_rate", "See ChordsDescriptors algorithm documentation");
  declareOutput(_chordsHistogram, "chords_histogram", "See ChordsDescriptors algorithm documentation");
  declareOutput(_chordsKey, "chords_key", "See ChordsDescriptors algorithm documentation");
  declareOutput(_chordsNumberRate, "chords_number_rate", "See ChordsDescriptors algorithm documentation");
  declareOutput(_chords, "chords_progression", "See ChordsDetection algorithm documentation");
  declareOutput(_chordsScale, "chords_scale", "See ChordsDetection algorithm documentation");
  declareOutput(_chordsStrength, "chords_strength", "See ChordsDetection algorithm documentation");
  declareOutput(_hpcp, "hpcp", "See HPCP algorithm documentation");
  declareOutput(_hpcpHighRes, "hpcp_highres", "See HPCP algorithm documentation");
  declareOutput(_key, "key_key", "See Key algorithm documentation");
  declareOutput(_scale, "key_scale", "See Key algorithm documentation");
  declareOutput(_keyStrength, "key_strength", "See Key algorithm documentation");

  createInnerNetwork();
}

TonalExtractor::~TonalExtractor() {
  delete _network;
}

void TonalExtractor::reset() {
  _network->reset();
}

void TonalExtractor::configure() {
  _tonalExtractor->configure(INHERIT("frameSize"), INHERIT("hopSize"), INHERIT("tuningFrequency"));
}

void TonalExtractor::createInnerNetwork() {
  _tonalExtractor = streaming::AlgorithmFactory::create("TonalExtractor");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _tonalExtractor->input("signal");

  _tonalExtractor->output("chords_changes_rate")  >>  PC(_pool, "chordsChangesRate");
  _tonalExtractor->output("chords_histogram")     >>  PC(_pool, "chordsHistogram");
  _tonalExtractor->output("chords_key")           >>  PC(_pool, "chordsKey");
  _tonalExtractor->output("chords_number_rate")   >>  PC(_pool, "chordsNumberRate");
  _tonalExtractor->output("chords_progression")   >>  PC(_pool, "chords");
  _tonalExtractor->output("chords_scale")         >>  PC(_pool, "chordsScale");
  _tonalExtractor->output("chords_strength")      >>  PC(_pool, "chordsStrength");
  _tonalExtractor->output("hpcp")                 >>  PC(_pool, "hpcp");
  _tonalExtractor->output("hpcp_highres")         >>  PC(_pool, "hpcpHighRes");
  _tonalExtractor->output("key_key")              >>  PC(_pool, "key");
  _tonalExtractor->output("key_scale")            >>  PC(_pool, "scale");
  _tonalExtractor->output("key_strength")         >>  PC(_pool, "keyStrength");

  _network = new scheduler::Network(_vectorInput);
}

void TonalExtractor::compute() {
  const vector<Real>& signal = _signal.get();
  _vectorInput->setVector(&signal);

  _network->run();

  vector<Real> &            chordsHistogram   = _chordsHistogram.get();
  Real &                    chordsChangesRate = _chordsChangesRate.get();
  string &                  chordsKey         = _chordsKey.get();
  Real &                    chordsNumberRate  = _chordsNumberRate.get();
  vector<string> &          chords            = _chords.get();
  string &                  chordsScale       = _chordsScale.get();
  vector<Real> &            chordsStrength    = _chordsStrength.get();
  vector<vector<Real> > &   hpcp              = _hpcp.get();
  vector<vector<Real> > &   hpcpHighRes       = _hpcpHighRes.get();
  string &                  key               = _key.get();
  string &                  scale             = _scale.get();
  Real &                    keyStrength       = _keyStrength.get();

  chordsHistogram   = _pool.value<vector<Real> > ("chordsHistogram");
  chordsChangesRate = _pool.value<Real> ("chordsChangesRate");
  chordsKey         = _pool.value<string> ("chordsKey");
  chordsNumberRate  = _pool.value<Real> ("chordsNumberRate");
  chords            = _pool.value<vector<string> > ("chords");
  chordsScale       = _pool.value<string> ("chordsScale");
  chordsStrength    = _pool.value<vector<Real> > ("chordsStrength");
  hpcp              = _pool.value<vector<vector<Real> > > ("hpcp");
  hpcpHighRes       = _pool.value<vector<vector<Real> > > ("hpcpHighRes");
  key               = _pool.value<string> ("key");
  scale             = _pool.value<string> ("scale");
  keyStrength       = _pool.value<Real> ("keyStrength");
}

} // namespace standard
} // namespace essentia
