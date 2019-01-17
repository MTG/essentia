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

#ifndef KEY_EXTRACTOR_H
#define KEY_EXTRACTOR_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "vectorinput.h"
#include "network.h"

namespace essentia {
namespace streaming {

class KeyExtractor : public AlgorithmComposite {
 protected:
  Algorithm *_frameCutter, *_windowing, *_spectrum, *_spectralPeaks, *_spectralWhitening, *_hpcpKey, *_key;
  scheduler::Network* _network;
  bool _configured;
  Real _sampleRate;
  int _frameSize;
  int _hopSize;
  Real _minFrequency;
  Real _maxFrequency;
  std::string _windowType;
  Real _spectralPeaksThreshold;
  int _maxPeaks;
  Real _tuningFrequency;
  int _hpcpSize;
  std::string _weightType;
  std::string _profileType;
  Real _pcpThreshold;
  bool _averageDetuningCorrection;

  SinkProxy<Real> _audio;
  SourceProxy<std::string> _keyKey;
  SourceProxy<std::string> _keyScale;
  SourceProxy<Real> _keyStrength;

 public:
  KeyExtractor();
  ~KeyExtractor();

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the framesize for computing tonal features", "(0,inf)", 4096);
    declareParameter("hopSize", "the hopsize for computing tonal features", "(0,inf)", 4096);
    declareParameter("windowType", "the window type, which can be 'hamming', 'hann', 'triangular', 'square' or 'blackmanharrisXX'", "{hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}", "hann");
    declareParameter("minFrequency", "min frequency to apply whitening to [Hz]", "(0,inf)", 25.0);
    declareParameter("maxFrequency", "max frequency to apply whitening to [Hz]", "(0,inf)", 3500.0);
    declareParameter("spectralPeaksThreshold", "the threshold for the spectral peaks", "(0,inf)", 0.0001);
    declareParameter("maximumSpectralPeaks", "the maximum number of spectral peaks", "(0,inf)", 60);
    declareParameter("hpcpSize", "the size of the output HPCP (must be a positive nonzero multiple of 12)", "[12,inf)", 12);
    declareParameter("weightType", "type of weighting function for determining frequency contribution", "{none,cosine,squaredCosine}", "cosine");
    declareParameter("tuningFrequency", "the tuning frequency of the input signal", "(0,inf)", 440.0);
    declareParameter("pcpThreshold", "pcp bins below this value are set to 0", "[0,1]", 0.2);
    declareParameter("averageDetuningCorrection", "shifts a pcp to the nearest tempered bin", "{true,false}", true);
    declareParameter("profileType", "the type of polyphic profile to use for correlation calculation", "{diatonic,krumhansl,temperley,weichai,tonictriad,temperley2005,thpcp,shaath,gomez,noland,faraldo,pentatonic,edmm,edma,bgate,braw}", "bgate");

  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
  }

  void configure();
  void createInnerNetwork();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

class KeyExtractor : public Algorithm {
 protected:
  Input<std::vector<Real> > _audio;
  Output<std::string> _key;
  Output<std::string> _scale;
  Output<Real> _strength;

  bool _configured;

  streaming::Algorithm* _keyExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  KeyExtractor();
  ~KeyExtractor();

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the framesize for computing tonal features", "(0,inf)", 4096);
    declareParameter("hopSize", "the hopsize for computing tonal features", "(0,inf)", 4096);
    declareParameter("windowType", "the window type, which can be 'hamming', 'hann', 'triangular', 'square' or 'blackmanharrisXX'", "{hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}", "hann");
    declareParameter("minFrequency", "min frequency to apply whitening to [Hz]", "(0,inf)", 25.0);
    declareParameter("maxFrequency", "max frequency to apply whitening to [Hz]", "(0,inf)", 3500.0);
    declareParameter("spectralPeaksThreshold", "the threshold for the spectral peaks", "(0,inf)", 0.0001);
    declareParameter("maximumSpectralPeaks", "the maximum number of spectral peaks", "(0,inf)", 60);
    declareParameter("hpcpSize", "the size of the output HPCP (must be a positive nonzero multiple of 12)", "[12,inf)", 12);
    declareParameter("weightType", "type of weighting function for determining frequency contribution", "{none,cosine,squaredCosine}", "cosine");
    declareParameter("tuningFrequency", "the tuning frequency of the input signal", "(0,inf)", 440.0);
    declareParameter("pcpThreshold", "pcp bins below this value are set to 0", "[0,1]", 0.2);
    declareParameter("averageDetuningCorrection", "shifts a pcp to the nearest tempered bin", "{true,false}", true);
    declareParameter("profileType", "the type of polyphic profile to use for correlation calculation", "{diatonic,krumhansl,temperley,weichai,tonictriad,temperley2005,thpcp,shaath,gomez,noland,faraldo,pentatonic,edmm,edma,bgate,braw}", "bgate");
  }

  void configure();
  void createInnerNetwork();
  void compute();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // KEY_EXTRACTOR_H
