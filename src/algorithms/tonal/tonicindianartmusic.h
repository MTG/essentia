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

#ifndef ESSENTIA_TONICINDIANARTMUSIC_H
#define ESSENTIA_TONICINDIANARTMUSIC_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class TonicIndianArtMusic : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _tonic;

  // Pre-processing
  Algorithm* _frameCutter;
  Algorithm* _windowing;

  // Spectral peaks
  Algorithm* _spectrum;
  Algorithm* _spectralPeaks;

  // Pitch salience contours
  Algorithm* _pitchSalienceFunction;
  Algorithm* _pitchSalienceFunctionPeaks;

  // tonic identification histogram processing
  Algorithm* _peakDetection;

  Real _referenceFrequency;
  Real _binResolution;
  Real _numberSaliencePeaks;
  Real _numberBins;

 public:
  TonicIndianArtMusic() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_tonic, "tonic", "the estimated tonic frequency [Hz]");

    // Pre-processing
    _frameCutter = AlgorithmFactory::create("FrameCutter");
    _windowing = AlgorithmFactory::create("Windowing");

    // Spectral peaks
    _spectrum = AlgorithmFactory::create("Spectrum");
    _spectralPeaks = AlgorithmFactory::create("SpectralPeaks");

    // Pitch salience contours
    _pitchSalienceFunction = AlgorithmFactory::create("PitchSalienceFunction");
    _pitchSalienceFunctionPeaks = AlgorithmFactory::create("PitchSalienceFunctionPeaks");

    // Peak detection of the histogram for tonic identification
    _peakDetection = AlgorithmFactory::create("PeakDetection");
  }

  ~TonicIndianArtMusic();

  void declareParameters() {
    // pre-processing
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size for computing pitch saliecnce", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size with which the pitch salience function was computed", "(0,inf)", 128);

    // pitch salience function
    declareParameter("binResolution", "salience function bin resolution [cents]", "(0,inf)", 10.0);
    declareParameter("referenceFrequency", "the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin", "(0,inf)", 55.0);
    declareParameter("magnitudeThreshold", "peak magnitude threshold (maximum allowed difference from the highest peak in dBs)", "[0,inf)",  40.0);
    declareParameter("magnitudeCompression", "magnitude compression parameter (=0 for maximum compression, =1 for no compression)", "(0,1]", 1.0);
    declareParameter("numberHarmonics", "number of considered hamonics", "[1,inf)", 20);
    declareParameter("harmonicWeight", "harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay)", "(0,1)", 0.8);

    // tonic identification using multipitch histogram
    declareParameter("numberSaliencePeaks", " number of top peaks of the salience function which should be considered for constructing histogram", "[1, 15]", 5);
    declareParameter("minTonicFrequency", "the minimum allowed tonic frequency [Hz]", "[0,inf)", 100.0);
    declareParameter("maxTonicFrequency", "the maximum allowed tonic frequency [Hz]", "[0,inf)", 375.0);
  }


  void compute();
  void configure();

  void reset() {
    _frameCutter->reset();
    _windowing->reset();
    _spectralPeaks->reset();
    _pitchSalienceFunctionPeaks->reset();
    _peakDetection->reset();
  }

  static const char* name;
  static const char* version;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_TONICINDIANARTMUSIC_H
