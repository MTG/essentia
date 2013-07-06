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

#ifndef ESSENTIA_ONSETRATE_H
#define ESSENTIA_ONSETRATE_H

#include "algorithmfactory.h"
#include "network.h"

namespace essentia {
namespace standard {

class OnsetRate : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _onsetTimes;
  Output<Real> _onsetRate;

  Real _sampleRate;
  int _frameSize;
  int _hopSize;
  Real _frameRate;
  int _zeroPadding;

  // Pre-processing
  Algorithm* _frameCutter;
  Algorithm* _windowing;

  // FFT
  Algorithm* _fft;
  Algorithm* _cartesian2polar;

  // Onsets
  Algorithm* _onsetHfc;
  Algorithm* _onsetComplex;
  Algorithm* _onsets;

public:
  OnsetRate() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_onsetTimes, "onsets", "the positions of detected onsets [s]");
    declareOutput(_onsetRate, "onsetRate", "the number of onsets per second");

    // Pre-processing
    _frameCutter = AlgorithmFactory::create("FrameCutter");
    _windowing = AlgorithmFactory::create("Windowing");

    // FFT
    _fft = AlgorithmFactory::create("FFT");
    _cartesian2polar = AlgorithmFactory::create("CartesianToPolar");

    // Onsets
    _onsetHfc = AlgorithmFactory::create("OnsetDetection");
    _onsetComplex = AlgorithmFactory::create("OnsetDetection");
    _onsets = AlgorithmFactory::create("Onsets");
  }

  ~OnsetRate();

  void declareParameters() {}

  void compute();
  void configure();

  void reset() {
    _frameCutter->reset();
    _onsets->reset();
    _onsetComplex->reset();
  }

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#include "pool.h"
#include "streamingalgorithmcomposite.h"

namespace essentia {
namespace streaming {

class OnsetRate : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _signal;

  Source<std::vector<Real> > _onsetTimes;
  Source<Real> _onsetRate;

  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _fft;
  Algorithm* _cart2polar;
  Algorithm* _onsetHfc;
  Algorithm* _onsetComplex;
  standard::Algorithm* _onsets;

  scheduler::Network* _network;

  Pool _pool;

  Real _sampleRate;
  int _frameSize;
  int _hopSize;
  Real _frameRate;
  int _zeroPadding;

  int _preferredBufferSize;

 public:
  OnsetRate();
  ~OnsetRate();

  void declareParameters() {};

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
    declareProcessStep(SingleShot(this));
  }

  void configure();
  AlgorithmStatus process();
  void reset();
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ONSETRATE_H
