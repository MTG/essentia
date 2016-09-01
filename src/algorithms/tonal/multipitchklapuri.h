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

#ifndef ESSENTIA_MULTIPITCHKLAPURI_H
#define ESSENTIA_MULTIPITCHKLAPURI_H

#include "algorithmfactory.h"
#include "network.h"

namespace essentia {
namespace standard {

class MultiPitchKlapuri : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<std::vector<std::vector<Real> > > _pitch;

  // Pre-processing
  Algorithm* _frameCutter;
  Algorithm* _windowing;

  // Spectral peaks and whitening
  Algorithm* _spectrum;
  Algorithm* _spectralPeaks;
  Algorithm* _spectralWhitening;

  // Pitch salience function
  Algorithm* _pitchSalienceFunction;
  Algorithm* _pitchSalienceFunctionPeaks;

  Real _sampleRate;
  int _frameSize;
  int _hopSize;
  int _zeroPaddingFactor;
  Real _referenceFrequency;
  Real _binResolution;
  int _numberHarmonics;
  int _numberHarmonicsMax;
  Real _centToHertzBase;
  int _binsInSemitone;
  int _binsInOctave;
  Real _referenceTerm;
  std::vector<Real> _centSpectrum;
  int _numberBins;

 public:
  MultiPitchKlapuri() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_pitch, "pitch", "the estimated pitch values [Hz]");

    // Pre-processing
    _frameCutter = AlgorithmFactory::create("FrameCutter");
    _windowing = AlgorithmFactory::create("Windowing");

    // Spectral peaks
    _spectrum = AlgorithmFactory::create("Spectrum");
    _spectralPeaks = AlgorithmFactory::create("SpectralPeaks");
    _spectralWhitening = AlgorithmFactory::create("SpectralWhitening");

    // Pitch salience contours
    _pitchSalienceFunction = AlgorithmFactory::create("PitchSalienceFunction");
    _pitchSalienceFunctionPeaks = AlgorithmFactory::create("PitchSalienceFunctionPeaks");
      
  }

  ~MultiPitchKlapuri();
    
  int frequencyToCentBin(Real frequency);
  Real getWeight(int centBin, int harmonicNumber);
    
  void declareParameters() {
      
    // pre-processing
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size for computing pitch saliecnce", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size with which the pitch salience function was computed", "(0,inf)", 128);

    // pitch salience function
    declareParameter("binResolution", "salience function bin resolution [cents]", "(0,inf)", 10.0);
    declareParameter("referenceFrequency", "the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin", "(0,inf)", 55.0);
    declareParameter("magnitudeThreshold", "spectral peak magnitude threshold (maximum allowed difference from the highest peak in dBs)", "[0,inf)",  40);
    declareParameter("magnitudeCompression", "magnitude compression parameter for the salience function (=0 for maximum compression, =1 for no compression)", "(0,1]", 1.0);
    declareParameter("numberHarmonics", "number of considered harmonics", "[1,inf)", 10);
    declareParameter("harmonicWeight", "harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay)", "(0,1)", 0.8);
      
    // pitch salience function peaks
    declareParameter("minFrequency", "the minimum allowed frequency for salience function peaks (ignore peaks below) [Hz]", "[0,inf)", 80.0);
    declareParameter("maxFrequency", "the maximum allowed frequency for salience function peaks (ignore peaks above) [Hz]", "[0,inf)", 1760.0); // max frequency in melodia
  }


  void compute();
  void configure();

  void reset() {
    _frameCutter->reset();
  }

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#endif 
