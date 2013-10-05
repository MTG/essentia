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

#include "lowlevelspectralextractor.h"
#include "algorithmfactory.h"
#include "essentiamath.h"
#include "poolstorage.h"
#include "copy.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;

const char* LowLevelSpectralExtractor::name = "LowLevelSpectralExtractor";
const char* LowLevelSpectralExtractor::description = DOC("This algorithm extracts all low level spectral features, which do not require an equal-loudness filter for their computation, from an audio signal");

LowLevelSpectralExtractor::LowLevelSpectralExtractor() : _configured(false) {

  // input:
  declareInput(_signal, "signal", "the input audio signal");

  // outputs:
  declareOutput(_bbands, "barkbands", "spectral energy at each bark band. See BarkBands alogithm");
  declareOutput(_bbandsKurtosis, "barkbands_kurtosis", "kurtosis from bark bands. See DistributionShape algorithm documentation");
  declareOutput(_bbandsSkewness, "barkbands_skewness", "skewness from bark bands. See DistributionShape algorithm documentation");
  declareOutput(_bbandsSpread, "barkbands_spread", "spread from barkbands. See DistributionShape algorithm documentation");
  declareOutput(_hfcValue, "hfc", "See HFC algorithm documentation");
  declareOutput(_mfccs, "mfcc", "See MFCC algorithm documentation");
  declareOutput(_pitchValue, "pitch", "See PitchYinFFT algorithm documentation");
  declareOutput(_pitchConfidence, "pitch_instantaneous_confidence", "See PitchYinFFT algorithm documentation");
  declareOutput(_pitchSalienceValue, "pitch_salience", "See PitchSalience algorithm documentation");
  declareOutput(_silence20, "silence_rate_20dB", "See SilenceRate algorithm documentation");
  declareOutput(_silence30, "silence_rate_30dB", "See SilenceRate algorithm documentation");
  declareOutput(_silence60, "silence_rate_60dB", "See SilenceRate algorithm documentation");
  declareOutput(_spectralComplexityValue, "spectral_complexity", "See Spectral algorithm documentation");
  declareOutput(_crestValue, "spectral_crest", "See Crest algorithm documentation");
  declareOutput(_decreaseValue, "spectral_decrease", "See Decrease algorithm documentation");
  declareOutput(_energyValue, "spectral_energy", "See Energy algorithm documentation");
  declareOutput(_ebandLow, "spectral_energyband_low", "Energy in band (20,150] Hz. See EnergyBand algorithm documentation");
  declareOutput(_ebandMidLow, "spectral_energyband_middle_low", "Energy in band (150,800] Hz.See EnergyBand algorithm documentation");
  declareOutput(_ebandMidHigh, "spectral_energyband_middle_high", "Energy in band (800,4000] Hz. See EnergyBand algorithm documentation");
  declareOutput(_ebandHigh, "spectral_energyband_high", "Energy in band (4000,20000] Hz. See EnergyBand algorithm documentation");
  declareOutput(_flatness, "spectral_flatness_db", "See flatnessDB algorithm documentation");
  declareOutput(_fluxValue, "spectral_flux", "See Flux algorithm documentation");
  declareOutput(_rmsValue, "spectral_rms", "See RMS algorithm documentation");
  declareOutput(_rolloffValue, "spectral_rolloff", "See RollOff algorithm documentation");
  declareOutput(_strongPeakValue, "spectral_strongpeak", "See StrongPeak algorithm documentation");
  declareOutput(_zeroCrossingRate, "zerocrossingrate", "See ZeroCrossingRate algorithm documentation");

  // sfx:
  declareOutput(_inharmonicityValue, "inharmonicity", "See Inharmonicity algorithm documentation");
  declareOutput(_tristimulusValue, "tristimulus", "See Tristimulus algorithm documentation");
  declareOutput(_odd2even, "oddtoevenharmonicenergyratio", "See OddToEvenHarmonicEnergyRatio algorithm documentation");

  // create network (instantiate algorithms)
  createInnerNetwork();

  // wire all this up!
  // connections:
  _signal                              >>  _frameCutter->input("signal");

  // connecting temporal descriptors
  _frameCutter->output("frame")        >>  _silenceRate->input("frame");
  _silenceRate->output("threshold_0")  >>  _silence20;
  _silenceRate->output("threshold_1")  >>  _silence30;
  _silenceRate->output("threshold_2")  >>  _silence60;

  _frameCutter->output("frame")        >>  _zcr->input("signal");
  _zcr->output("zeroCrossingRate")     >>  _zeroCrossingRate;

  // connecting spectral descriptors
  _frameCutter->output("frame")        >>  _windowing->input("frame");
  _windowing->output("frame")          >>  _spectrum->input("frame");

  _spectrum->output("spectrum")        >>  _mfcc->input("spectrum");
  _mfcc->output("mfcc")                >>  _mfccs;
  _mfcc->output("bands")               >>  NOWHERE;

  _spectrum->output("spectrum")        >>  _energy->input("array");
  _spectrum->output("spectrum")        >>  _energyBand_0->input("spectrum");
  _spectrum->output("spectrum")        >>  _energyBand_1->input("spectrum");
  _spectrum->output("spectrum")        >>  _energyBand_2->input("spectrum");
  _spectrum->output("spectrum")        >>  _energyBand_3->input("spectrum");
  _energy->output("energy")            >>  _energyValue;
  _energyBand_0->output("energyBand")  >>  _ebandLow;
  _energyBand_1->output("energyBand")  >>  _ebandMidLow;
  _energyBand_2->output("energyBand")  >>  _ebandMidHigh;
  _energyBand_3->output("energyBand")  >>  _ebandHigh;

  _spectrum->output("spectrum")        >>  _hfc->input("spectrum");
  _hfc->output("hfc")                  >>  _hfcValue;

  _spectrum->output("spectrum")        >>  _rms->input("array");
  _rms->output("rms")                  >>  _rmsValue;

  _spectrum->output("spectrum")        >>  _flux->input("spectrum");
  _flux->output("flux")                >>  _fluxValue;

  _spectrum->output("spectrum")        >>  _rollOff->input("spectrum");
  _rollOff->output("rollOff")          >>  _rolloffValue;

  _spectrum->output("spectrum")        >>  _strongPeak->input("spectrum");
  _strongPeak->output("strongPeak")    >>  _strongPeakValue;

  _spectrum->output("spectrum")        >>  _square->input("array");
  _square->output("array")      >>  _decrease->input("array");

  _spectrum->output("spectrum")                      >>  _spectralComplexity->input("spectrum");
  _spectralComplexity->output("spectralComplexity")  >>  _spectralComplexityValue;

  _spectrum->output("spectrum")               >>  _pitchDetection->input("spectrum");
  _pitchDetection->output("pitch")            >>  _pitchValue;
  _pitchDetection->output("pitchConfidence")  >>  _pitchConfidence;
  _spectrum->output("spectrum")               >>  _pitchSalience->input("spectrum");
  _pitchSalience->output("pitchSalience")     >>  _pitchSalienceValue;


  _spectrum->output("spectrum")               >>  _barkBands->input("spectrum");
  _barkBands->output("bands")                 >>  _bbands;
  _barkBands->output("bands")                 >>  _crest->input("array");
  _barkBands->output("bands")                 >>  _flatnessdb->input("array");
  _barkBands->output("bands")                 >>  _centralMoments->input("array");
  _crest->output("crest")                     >>  _crestValue;
  _decrease->output("decrease")               >>  _decreaseValue;
  _flatnessdb->output("flatnessDB")           >>  _flatness;

  _centralMoments->output("centralMoments")   >>  _distributionShape->input("centralMoments");
  _distributionShape->output("kurtosis")      >>  _bbandsKurtosis;
  _distributionShape->output("skewness")      >>  _bbandsSkewness;
  _distributionShape->output("spread")        >>  _bbandsSpread;

  // sfx:
  _pitchDetection->output("pitch")                >>  _harmonicPeaks->input("pitch");
  _spectrum->output("spectrum")                   >>  _spectralPeaks->input("spectrum");
  _spectralPeaks->output("frequencies")           >>  _harmonicPeaks->input("frequencies");
  _spectralPeaks->output("magnitudes")            >>  _harmonicPeaks->input("magnitudes");

  _harmonicPeaks->output("harmonicMagnitudes")    >>  _tristimulus->input("magnitudes");
  _harmonicPeaks->output("harmonicFrequencies")   >>  _tristimulus->input("frequencies");
  _harmonicPeaks->output("harmonicMagnitudes")    >>  _oddToEvenHarmonicEnergyRatio->input("magnitudes");
  _harmonicPeaks->output("harmonicFrequencies")   >>  _oddToEvenHarmonicEnergyRatio->input("frequencies");
  _harmonicPeaks->output("harmonicMagnitudes")    >>  _inharmonicity->input("magnitudes");
  _harmonicPeaks->output("harmonicFrequencies")   >>  _inharmonicity->input("frequencies");

  _inharmonicity->output("inharmonicity")         >>  _inharmonicityValue;
  _tristimulus->output("tristimulus")             >>  _tristimulusValue;
  _oddToEvenHarmonicEnergyRatio->output("oddToEvenHarmonicEnergyRatio")  >>  _odd2even;

  _network = new scheduler::Network(_frameCutter);
}

void LowLevelSpectralExtractor::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _barkBands          = factory.create("BarkBands",
                                       "numberBands", 27);
  _centralMoments     = factory.create("CentralMoments",
                                       "range", 26);
  _crest              = factory.create("Crest");
  _decrease           = factory.create("Decrease");
  _distributionShape  = factory.create("DistributionShape");
  _energyBand_0       = factory.create("EnergyBand",
                                       "startCutoffFrequency", 20,   "stopCutoffFrequency", 150);
  _energyBand_1       = factory.create("EnergyBand",
                                       "startCutoffFrequency", 150,  "stopCutoffFrequency", 800);
  _energyBand_2       = factory.create("EnergyBand",
                                       "startCutoffFrequency", 800,  "stopCutoffFrequency", 4000);
  _energyBand_3       = factory.create("EnergyBand",
                                       "startCutoffFrequency", 4000, "stopCutoffFrequency", 20000);
  _energy             = factory.create("Energy");
  _flatnessdb         = factory.create("FlatnessDB");
  _flux               = factory.create("Flux");
  _frameCutter        = factory.create("FrameCutter");
  _hfc                = factory.create("HFC");
  _harmonicPeaks      = factory.create("HarmonicPeaks");
  _inharmonicity      = factory.create("Inharmonicity");
  _mfcc               = factory.create("MFCC");
  _oddToEvenHarmonicEnergyRatio = factory.create("OddToEvenHarmonicEnergyRatio");
  _pitchDetection     = factory.create("PitchYinFFT");
  _pitchSalience      = factory.create("PitchSalience");
  _rms                = factory.create("RMS");
  _rollOff            = factory.create("RollOff");
  _silenceRate        = factory.create("SilenceRate");
  _spectralComplexity = factory.create("SpectralComplexity",
                                       "magnitudeThreshold", 0.005);
  _spectralPeaks      = factory.create("SpectralPeaks");
  _spectrum           = factory.create("Spectrum");
  _strongPeak         = factory.create("StrongPeak");
  _tristimulus        = factory.create("Tristimulus");
  _square             = factory.create("UnaryOperator",
                                       "type", "square");
  _windowing          = factory.create("Windowing",
                                       "type", "blackmanharris62");
  _zcr                = factory.create("ZeroCrossingRate");

  Real thresholds_dB[] = { -20, -30, -60 };
  vector<Real> thresholds(ARRAY_SIZE(thresholds_dB));
  for (int i=0; i<(int)thresholds.size(); i++) {
    thresholds[i] = db2lin(thresholds_dB[i]/2.0);
  }
  _silenceRate->configure("thresholds", thresholds);

}

void LowLevelSpectralExtractor::configure() {
  int frameSize   = parameter("frameSize").toInt();
  int hopSize     = parameter("hopSize").toInt();
  Real sampleRate = parameter("sampleRate").toReal();

  _decrease->configure("range", 0.5 * sampleRate);
  _frameCutter->configure("silentFrames", "noise", "hopSize", hopSize, "frameSize", frameSize);
  _pitchDetection->configure("frameSize", frameSize);
  _spectralPeaks->configure("orderBy", "frequency", "minFrequency", sampleRate/Real(frameSize));
}

LowLevelSpectralExtractor::~LowLevelSpectralExtractor() {
  clearAlgos();
}

void LowLevelSpectralExtractor::clearAlgos() {
  if (!_configured) return;
  delete _network;
}

namespace essentia {
namespace standard {

const char* LowLevelSpectralExtractor::name = "LowLevelSpectralExtractor";
const char* LowLevelSpectralExtractor::description = DOC("This algorithm extracts all low level spectral features from an audio signal");

LowLevelSpectralExtractor::LowLevelSpectralExtractor() {
  declareInput(_signal, "signal", "the audio input signal");
  declareOutput(_barkBands, "barkbands", "spectral energy at each bark band. See BarkBands alogithm");
  declareOutput(_kurtosis, "barkbands_kurtosis", "kurtosis from bark bands. See DistributionShape algorithm documentation");
  declareOutput(_skewness, "barkbands_skewness", "skewness from bark bands. See DistributionShape algorithm documentation");
  declareOutput(_spread, "barkbands_spread", "spread from barkbands. See DistributionShape algorithm documentation");
  declareOutput(_hfc, "hfc", "See HFC algorithm documentation");
  declareOutput(_mfcc, "mfcc", "See MFCC algorithm documentation");
  declareOutput(_pitch, "pitch", "See PitchYinFFT algorithm documentation");
  declareOutput(_pitchConfidence, "pitch_instantaneous_confidence", "See PitchYinFFT algorithm documentation");
  declareOutput(_pitchSalience, "pitch_salience", "See PitchSalience algorithm documentation");
  declareOutput(_threshold_0, "silence_rate_20dB", "See SilenceRate algorithm documentation");
  declareOutput(_threshold_1, "silence_rate_30dB", "See SilenceRate algorithm documentation");
  declareOutput(_threshold_2, "silence_rate_60dB", "See SilenceRate algorithm documentation");
  declareOutput(_spectralComplexity, "spectral_complexity", "See Spectral algorithm documentation");
  declareOutput(_crest, "spectral_crest", "See Crest algorithm documentation");
  declareOutput(_decrease, "spectral_decrease", "See Decrease algorithm documentation");
  declareOutput(_energy, "spectral_energy", "See Energy algorithm documentation");
  declareOutput(_energyBand_0, "spectral_energyband_low", "Energy in band (20,150] Hz. See EnergyBand algorithm documentation");
  declareOutput(_energyBand_1, "spectral_energyband_middle_low", "Energy in band (150,800] Hz.See EnergyBand algorithm documentation");
  declareOutput(_energyBand_2, "spectral_energyband_middle_high", "Energy in band (800,4000] Hz. See EnergyBand algorithm documentation");
  declareOutput(_energyBand_3, "spectral_energyband_high", "Energy in band (4000,20000] Hz. See EnergyBand algorithm documentation");
  declareOutput(_flatnessdb, "spectral_flatness_db", "See flatnessDB algorithm documentation");
  declareOutput(_flux, "spectral_flux", "See Flux algorithm documentation");
  declareOutput(_rms, "spectral_rms", "See RMS algorithm documentation");
  declareOutput(_rollOff, "spectral_rolloff", "See RollOff algorithm documentation");
  declareOutput(_strongPeak, "spectral_strongpeak", "See StrongPeak algorithm documentation");
  declareOutput(_zeroCrossingRate, "zerocrossingrate", "See ZeroCrossingRate algorithm documentation");
  // sfx:
  declareOutput(_inharmonicity, "inharmonicity", "See Inharmonicity algorithm documentation");
  declareOutput(_tristimulus, "tristimulus", "See Tristimulus algorithm documentation");
  declareOutput(_oddToEvenHarmonicEnergyRatio, "oddtoevenharmonicenergyratio", "See OddToEvenHarmonicEnergyRatio algorithm documentation");

  _lowLevelExtractor = streaming::AlgorithmFactory::create("LowLevelSpectralExtractor");
  _vectorInput = new streaming::VectorInput<Real>();
  createInnerNetwork();
}

LowLevelSpectralExtractor::~LowLevelSpectralExtractor() {
  delete _network;
}

void LowLevelSpectralExtractor::reset() {
  _network->reset();
}

void LowLevelSpectralExtractor::configure() {
  _lowLevelExtractor->configure(INHERIT("frameSize"), INHERIT("hopSize"), INHERIT("sampleRate"));
}

void LowLevelSpectralExtractor::createInnerNetwork() {
  streaming::connect(*_vectorInput, _lowLevelExtractor->input("signal"));
  streaming::connect(_lowLevelExtractor->output("barkbands"), _pool, "barkbands");
  streaming::connect(_lowLevelExtractor->output("barkbands_kurtosis"), _pool, "kurtosis");
  streaming::connect(_lowLevelExtractor->output("barkbands_skewness"), _pool, "skewness");
  streaming::connect(_lowLevelExtractor->output("barkbands_spread"), _pool, "spread");
  streaming::connect(_lowLevelExtractor->output("hfc"), _pool, "hfc");
  streaming::connect(_lowLevelExtractor->output("mfcc"), _pool, "mfcc");
  streaming::connect(_lowLevelExtractor->output("pitch"), _pool, "pitch");
  streaming::connect(_lowLevelExtractor->output("pitch_instantaneous_confidence"), _pool, "pitchConfidence");
  streaming::connect(_lowLevelExtractor->output("pitch_salience"), _pool, "pitchSalience");
  streaming::connect(_lowLevelExtractor->output("silence_rate_20dB"), _pool, "silence_rate_20dB");
  streaming::connect(_lowLevelExtractor->output("silence_rate_30dB"), _pool, "silence_rate_30dB");
  streaming::connect(_lowLevelExtractor->output("silence_rate_60dB"), _pool, "silence_rate_60dB");
  streaming::connect(_lowLevelExtractor->output("spectral_complexity"), _pool, "spectralComplexity");
  streaming::connect(_lowLevelExtractor->output("spectral_crest"), _pool, "crest");
  streaming::connect(_lowLevelExtractor->output("spectral_decrease"), _pool, "decrease");
  streaming::connect(_lowLevelExtractor->output("spectral_energy"), _pool, "energy");
  streaming::connect(_lowLevelExtractor->output("spectral_energyband_low"), _pool, "energyBand_0");
  streaming::connect(_lowLevelExtractor->output("spectral_energyband_middle_low"), _pool, "energyBand_1");
  streaming::connect(_lowLevelExtractor->output("spectral_energyband_middle_high"), _pool, "energyBand_2");
  streaming::connect(_lowLevelExtractor->output("spectral_energyband_high"), _pool, "energyBand_3");
  streaming::connect(_lowLevelExtractor->output("spectral_flatness_db"), _pool, "flatnessdb");
  streaming::connect(_lowLevelExtractor->output("spectral_flux"), _pool, "flux");
  streaming::connect(_lowLevelExtractor->output("spectral_rms"), _pool, "rms");
  streaming::connect(_lowLevelExtractor->output("spectral_rolloff"), _pool, "rollOff");
  streaming::connect(_lowLevelExtractor->output("spectral_strongpeak"), _pool, "strongPeak");
  streaming::connect(_lowLevelExtractor->output("zerocrossingrate"), _pool, "zeroCrossingRate");

  streaming::connect(_lowLevelExtractor->output("inharmonicity"), _pool, "inharmonicity");
  streaming::connect(_lowLevelExtractor->output("tristimulus"), _pool, "tristimulus");
  streaming::connect(_lowLevelExtractor->output("oddtoevenharmonicenergyratio"), _pool, "oddToEvenHarmonicEnergyRatio");

  _network = new scheduler::Network(_vectorInput);
}

void LowLevelSpectralExtractor::compute() {
  const vector<Real>& signal = _signal.get();
  _vectorInput->setVector(&signal);

  _network->run();

  vector<vector<Real> > & barkBands = _barkBands.get();
  vector<Real> & kurtosis = _kurtosis.get();
  vector<Real> & skewness = _skewness.get();
  vector<Real> & spread = _spread.get();
  vector<Real> & hfc = _hfc.get();
  vector<vector<Real> > & mfcc = _mfcc.get();
  vector<Real> & pitch = _pitch.get();
  vector<Real> & pitchConfidence = _pitchConfidence.get();
  vector<Real> & pitchSalience = _pitchSalience.get();
  vector<Real> & threshold_0 = _threshold_0.get();
  vector<Real> & threshold_1 = _threshold_1.get();
  vector<Real> & threshold_2 = _threshold_2.get();
  vector<Real> & spectralComplexity = _spectralComplexity.get();
  vector<Real> & crest = _crest.get();
  vector<Real> & decrease = _decrease.get();
  vector<Real> & energy = _energy.get();
  vector<Real> & energyBand_0 = _energyBand_0.get();
  vector<Real> & energyBand_1 = _energyBand_1.get();
  vector<Real> & energyBand_2 = _energyBand_2.get();
  vector<Real> & energyBand_3 = _energyBand_3.get();
  vector<Real> & flatnessdb = _flatnessdb.get();
  vector<Real> & flux = _flux.get();
  vector<Real> & rms = _rms.get();
  vector<Real> & rollOff = _rollOff.get();
  vector<Real> & strongPeak = _strongPeak.get();
  vector<Real> & zeroCrossingRate = _zeroCrossingRate.get();
  vector<Real> & inharmonicity = _inharmonicity.get();
  vector<vector<Real> > & tristimulus = _tristimulus.get();
  vector<Real> & oddToEvenHarmonicEnergyRatio = _oddToEvenHarmonicEnergyRatio.get();

  barkBands =          _pool.value<vector<vector<Real> > >("barkbands");
  kurtosis =           _pool.value<vector<Real> >("kurtosis");
  skewness =           _pool.value<vector<Real> >("skewness");
  spread =             _pool.value<vector<Real> >("spread");
  hfc =                _pool.value<vector<Real> >("hfc");
  mfcc =               _pool.value<vector<vector<Real> > >("mfcc");
  pitch =              _pool.value<vector<Real> >("pitch");
  pitchConfidence =    _pool.value<vector<Real> >("pitchConfidence");
  pitchSalience =      _pool.value<vector<Real> >("pitchSalience");
  threshold_0 =        _pool.value<vector<Real> >("silence_rate_20dB");
  threshold_1 =        _pool.value<vector<Real> >("silence_rate_30dB");
  threshold_2 =        _pool.value<vector<Real> >("silence_rate_60dB");
  spectralComplexity = _pool.value<vector<Real> >("spectralComplexity");
  crest =              _pool.value<vector<Real> >("crest");
  decrease =           _pool.value<vector<Real> >("decrease");
  energy =             _pool.value<vector<Real> >("energy");
  energyBand_0 =       _pool.value<vector<Real> >("energyBand_0");
  energyBand_1 =       _pool.value<vector<Real> >("energyBand_1");
  energyBand_2 =       _pool.value<vector<Real> >("energyBand_2");
  energyBand_3 =       _pool.value<vector<Real> >("energyBand_3");
  flatnessdb =         _pool.value<vector<Real> >("flatnessdb");
  flux =               _pool.value<vector<Real> >("flux");
  rms =                _pool.value<vector<Real> >("rms");
  rollOff =            _pool.value<vector<Real> >("rollOff");
  strongPeak =         _pool.value<vector<Real> >("strongPeak");
  zeroCrossingRate =   _pool.value<vector<Real> >("zeroCrossingRate");
  inharmonicity =      _pool.value<vector<Real> >("inharmonicity");
  tristimulus =        _pool.value<vector<vector<Real> > >("tristimulus");
  oddToEvenHarmonicEnergyRatio = _pool.value<vector<Real> >("oddToEvenHarmonicEnergyRatio");
}

} // namespace standard
} // namespace essentia
