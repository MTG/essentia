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

#include "lowlevelspectraleqloudextractor.h"
#include "algorithmfactory.h"
#include "poolstorage.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* LowLevelSpectralEqloudExtractor::name = "LowLevelSpectralEqloudExtractor";
const char* LowLevelSpectralEqloudExtractor::description = DOC("This algorithm extracts a set of level spectral features for which it is recommended to apply a preliminary equal-loudness filter over an input audio signal (according to the internal evaluations conducted at Music Technology Group). To this end, you are expected to provide the output of EqualLoudness algorithm as an input for this algorithm. Still, you are free to provide an unprocessed audio input in the case you want to compute these features without equal-loudness filter.\n"
"\n"
"Note that at present we do not dispose any reference to justify the necessity of equal-loudness filter. Our recommendation is grounded on internal evaluations conducted at Music Technology Group that have shown the increase in numeric robustness as a function of the audio encoders used (mp3, ogg, ...) for these features.");

LowLevelSpectralEqloudExtractor::LowLevelSpectralEqloudExtractor() : _configured(false) {

  declareInput(_signal, "signal", "the input audio signal");

  declareOutput(_scentroid, "spectral_centroid", "See Centroid algorithm documentation");
  declareOutput(_dissonanceValue, "dissonance", "See Dissonance algorithm documentation");
  declareOutput(_sccontrast, "sccoeffs", "See SpectralContrast algorithm documentation");
  declareOutput(_scvalleys, "scvalleys", "See SpectralContrast algorithm documentation");
  declareOutput(_kurtosis, "spectral_kurtosis", "See DistributionShape algorithm documentation");
  declareOutput(_skewness, "spectral_skewness", "See DistributionShape algorithm documentation");
  declareOutput(_spread, "spectral_spread", "See DistributionShape algorithm documentation");

  createInnerNetwork();
}

void LowLevelSpectralEqloudExtractor::createInnerNetwork() {
  // instantiate algorithms
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter        = factory.create("FrameCutter");
  _windowing          = factory.create("Windowing",
                                       "type", "blackmanharris62");
  _spectrum           = factory.create("Spectrum");
  _centralMoments     = factory.create("CentralMoments");
  _square             = factory.create("UnaryOperator",
                                       "type", "square");
  _centroid           = factory.create("Centroid");
  _dissonance         = factory.create("Dissonance");
  _distributionShape  = factory.create("DistributionShape");
  _spectralContrast   = factory.create("SpectralContrast");
  _spectralPeaks      = factory.create("SpectralPeaks",
                                       "orderBy", "frequency");

  // connect all algorithms
  _signal                                        >>  _frameCutter->input("signal");
  _frameCutter->output("frame")                  >>  _windowing->input("frame");
  _windowing->output("frame")                    >>  _spectrum->input("frame");

  _spectrum->output("spectrum")                  >>  _square->input("array");
  _square->output("array")                       >>  _centroid->input("array");
  _centroid->output("centroid")                  >>  _scentroid;

  _spectrum->output("spectrum")                  >>  _spectralContrast->input("spectrum");
  _spectralContrast->output("spectralContrast")  >>  _sccontrast;
  _spectralContrast->output("spectralValley")    >>  _scvalleys;

  _spectrum->output("spectrum")                  >>  _centralMoments->input("array");
  _centralMoments->output("centralMoments")      >>  _distributionShape->input("centralMoments");
  _distributionShape->output("kurtosis")         >>  _kurtosis;
  _distributionShape->output("skewness")         >>  _skewness;
  _distributionShape->output("spread")           >>  _spread;

  _spectrum->output("spectrum")                  >>  _spectralPeaks->input("spectrum");
  _spectralPeaks->output("magnitudes")           >>  _dissonance->input("magnitudes");
  _spectralPeaks->output("frequencies")          >>  _dissonance->input("frequencies");
  _dissonance->output("dissonance")              >>  _dissonanceValue;

  _network = new scheduler::Network(_frameCutter);
}

void LowLevelSpectralEqloudExtractor::configure() {
  int frameSize = parameter("frameSize").toInt();
  int hopSize = parameter("hopSize").toInt();
  Real sampleRate = parameter("sampleRate").toReal();
  Real halfSampleRate = 0.5*sampleRate;

  _centralMoments->configure("range", halfSampleRate);
  _centroid->configure("range", halfSampleRate);
  _frameCutter->configure("silentFrames", "noise", "hopSize", hopSize, "frameSize", frameSize);
  _spectralContrast->configure("neighbourRatio", 0.4, "frameSize", frameSize,
                               "staticDistribution", 0.15, "numberBands", 6,
                               "lowFrequencyBound", 20, "sampleRate", sampleRate,
                               "highFrequencyBound", 11000);
  _configured = true;
}


LowLevelSpectralEqloudExtractor::~LowLevelSpectralEqloudExtractor() {
  delete _network;
}

} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

const char* LowLevelSpectralEqloudExtractor::name = "LowLevelSpectralEqloudExtractor";
const char* LowLevelSpectralEqloudExtractor::description = DOC("This algorithm extracts a set of level spectral features for which it is recommended to apply a preliminary equal-loudness filter over an input audio signal (according to the internal evaluations conducted at Music Technology Group). To this end, you are expected to provide the output of EqualLoudness algorithm as an input for this algorithm. Still, you are free to provide an unprocessed audio input in the case you want to compute these features without equal-loudness filter.\n"
"\n"
"Note that at present we do not dispose any reference to justify the necessity of equal-loudness filter. Our recommendation is grounded on internal evaluations conducted at Music Technology Group that have shown the increase in numeric robustness as a function of the audio encoders used (mp3, ogg, ...) for these features.");

LowLevelSpectralEqloudExtractor::LowLevelSpectralEqloudExtractor() {
  declareInput(_signal,      "signal", "the input audio signal");
  declareOutput(_dissonance, "dissonance",        "See Dissonance algorithm documentation");
  declareOutput(_sccoeffs,   "sccoeffs",          "See SpectralContrast algorithm documentation");
  declareOutput(_scvalleys,  "scvalleys",         "See SpectralContrast algorithm documentation");
  declareOutput(_centroid,   "spectral_centroid", "See Centroid algorithm documentation");
  declareOutput(_kurtosis,   "spectral_kurtosis", "See DistributionShape algorithm documentation");
  declareOutput(_skewness,   "spectral_skewness", "See DistributionShape algorithm documentation");
  declareOutput(_spread,     "spectral_spread",   "See DistributionShape algorithm documentation");

  createInnerNetwork();
}

LowLevelSpectralEqloudExtractor::~LowLevelSpectralEqloudExtractor() {
  delete _network;
}

void LowLevelSpectralEqloudExtractor::configure() {
  _lowlevelExtractor->configure(INHERIT("frameSize"), INHERIT("hopSize"), INHERIT("sampleRate"));
}

void LowLevelSpectralEqloudExtractor::reset() {
  delete _network;
}

void LowLevelSpectralEqloudExtractor::createInnerNetwork() {
  _lowlevelExtractor = streaming::AlgorithmFactory::create("LowLevelSpectralEqloudExtractor");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _lowlevelExtractor->input("signal");

  _lowlevelExtractor->output("dissonance")         >>  PC(_pool, "internal.dissonance");
  _lowlevelExtractor->output("sccoeffs")           >>  PC(_pool, "internal.sccoeffs");
  _lowlevelExtractor->output("scvalleys")          >>  PC(_pool, "internal.scvalleys");
  _lowlevelExtractor->output("spectral_centroid")  >>  PC(_pool, "internal.centroid");
  _lowlevelExtractor->output("spectral_kurtosis")  >>  PC(_pool, "internal.kurtosis");
  _lowlevelExtractor->output("spectral_skewness")  >>  PC(_pool, "internal.skewness");
  _lowlevelExtractor->output("spectral_spread")    >>  PC(_pool, "internal.spread");

  _network = new scheduler::Network(_vectorInput);
}

void LowLevelSpectralEqloudExtractor::compute() {
  const vector<Real>& signal = _signal.get();
  _vectorInput->setVector(&signal);

  _network->run();

  vector<Real> & dissonance = _dissonance.get();
  vector<vector<Real> > & sccoeffs = _sccoeffs.get();
  vector<vector<Real> > & scvalleys = _scvalleys.get();
  vector<Real> & centroid = _centroid.get();
  vector<Real> & kurtosis = _kurtosis.get();
  vector<Real> & skewness = _skewness.get();
  vector<Real> & spread = _spread.get();

  dissonance = _pool.value<vector<Real> >("internal.dissonance");
  sccoeffs   = _pool.value<vector<vector<Real> > >("internal.sccoeffs");
  scvalleys  = _pool.value<vector<vector<Real> > >("internal.scvalleys");
  centroid   = _pool.value<vector<Real> >("internal.centroid");
  kurtosis   = _pool.value<vector<Real> >("internal.kurtosis");
  skewness   = _pool.value<vector<Real> >("internal.skewness");
  spread     = _pool.value<vector<Real> >("internal.spread");
}

} // namespace standard
} // namespace essentia

