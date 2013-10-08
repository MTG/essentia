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

#ifndef LOW_LEVEL_SPECTRAL_EXTRACTOR_H
#define LOW_LEVEL_SPECTRAL_EXTRACTOR_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "vectorinput.h"
#include "network.h"

namespace essentia {
namespace streaming {

class LowLevelSpectralExtractor : public AlgorithmComposite {
 protected:
  SinkProxy<Real> _signal;

  SourceProxy<std::vector<Real> > _bbands;
  SourceProxy<Real> _bbandsKurtosis;
  SourceProxy<Real> _bbandsSkewness;
  SourceProxy<Real> _bbandsSpread;
  SourceProxy<Real> _hfcValue;
  SourceProxy<std::vector<Real> > _mfccs;
  SourceProxy<Real> _pitchValue;
  SourceProxy<Real> _pitchConfidence;
  SourceProxy<Real> _pitchSalienceValue;
  SourceProxy<Real> _silence20;
  SourceProxy<Real> _silence30;
  SourceProxy<Real> _silence60;
  SourceProxy<Real> _spectralComplexityValue;
  SourceProxy<Real> _crestValue;
  SourceProxy<Real> _decreaseValue;
  SourceProxy<Real> _energyValue;
  SourceProxy<Real> _ebandLow;
  SourceProxy<Real> _ebandMidLow;
  SourceProxy<Real> _ebandMidHigh;
  SourceProxy<Real> _ebandHigh;
  SourceProxy<Real> _flatness;
  SourceProxy<Real> _fluxValue;
  SourceProxy<Real> _rmsValue;
  SourceProxy<Real> _rolloffValue;
  SourceProxy<Real> _strongPeakValue;
  SourceProxy<Real> _zeroCrossingRate;

  SourceProxy<Real> _inharmonicityValue;
  SourceProxy<std::vector<Real> > _tristimulusValue;
  SourceProxy<Real> _odd2even;

  Algorithm *_barkBands, *_centralMoments, *_crest, *_decrease,
            *_distributionShape, *_energyBand_0, *_energyBand_1,
            *_energyBand_2, *_energyBand_3, *_energy, *_flatnessdb,
            *_flux, *_frameCutter, *_hfc, *_harmonicPeaks, *_inharmonicity,
            *_mfcc, *_oddToEvenHarmonicEnergyRatio, *_pitchDetection,
            *_pitchSalience, *_rms, *_rollOff, *_silenceRate, *_spectralComplexity,
            *_spectralPeaks, *_spectrum, *_strongPeak, *_tristimulus,
            *_square, *_windowing, *_zcr;

  scheduler::Network* _network;

  bool _configured;
  void clearAlgos();

 public:
  LowLevelSpectralExtractor();
  ~LowLevelSpectralExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the frame size for computing low level features", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size for computing low level features", "(0,inf)", 1024);
    declareParameter("sampleRate", "the audio sampling rate", "(0,inf)", 44100.0);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
  }

  void configure();
  void createInnerNetwork();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

class LowLevelSpectralExtractor : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<std::vector<Real> > > _barkBands;
  Output<std::vector<Real> > _kurtosis;
  Output<std::vector<Real> > _skewness;
  Output<std::vector<Real> > _spread;
  Output<std::vector<Real> > _hfc;
  Output<std::vector<std::vector<Real> > > _mfcc;
  Output<std::vector<Real> > _pitch;
  Output<std::vector<Real> > _pitchConfidence;
  Output<std::vector<Real> > _pitchSalience;
  Output<std::vector<Real> > _threshold_0;
  Output<std::vector<Real> > _threshold_1;
  Output<std::vector<Real> > _threshold_2;
  Output<std::vector<Real> > _spectralComplexity;
  Output<std::vector<Real> > _crest;
  Output<std::vector<Real> > _decrease;
  Output<std::vector<Real> > _energy;
  Output<std::vector<Real> > _energyBand_0;
  Output<std::vector<Real> > _energyBand_1;
  Output<std::vector<Real> > _energyBand_2;
  Output<std::vector<Real> > _energyBand_3;
  Output<std::vector<Real> > _flatnessdb;
  Output<std::vector<Real> > _flux;
  Output<std::vector<Real> > _rms;
  Output<std::vector<Real> > _rollOff;
  Output<std::vector<Real> > _strongPeak;
  Output<std::vector<Real> > _zeroCrossingRate;
  Output<std::vector<Real> > _inharmonicity;
  Output<std::vector<std::vector<Real> > > _tristimulus;
  Output<std::vector<Real> > _oddToEvenHarmonicEnergyRatio;

  bool _configured;

  streaming::Algorithm* _lowLevelExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  LowLevelSpectralExtractor();
  ~LowLevelSpectralExtractor();

  void declareParameters() {
    declareParameter("frameSize", "the frame size for computing low level features", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size for computing low level features", "(0,inf)", 1024);
    declareParameter("sampleRate", "the audio sampling rate", "(0,inf)", 44100.0);
  }

  void configure();
  void createInnerNetwork();
  void compute();
  void reset();

  static const char* name;
  static const char* description;
};

}
}

#endif
