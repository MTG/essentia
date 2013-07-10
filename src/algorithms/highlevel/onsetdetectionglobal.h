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

#ifndef ESSENTIA_ONSETDETECTIONGLOBAL_H
#define ESSENTIA_ONSETDETECTIONGLOBAL_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class OnsetDetectionGlobal : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _onsetDetections;

  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _spectrum;
  Algorithm* _fft;
  Algorithm* _cartesian2polar;
  Algorithm* _movingAverage;
  Algorithm* _erbbands;
  Algorithm* _autocorrelation;

  std::string _method;

  std::vector<Real> _frame;
  std::vector<Real> _frameWindowed;

  int _minFrequencyBin;
  int _maxFrequencyBin;
  int _numberFFTBins;
  int _bufferSize;
  int _histogramSize;
  std::vector<Real> _weights;
  std::vector<Real> _rweights;

  // beat emphasis function
  int _numberERBBands;
  static const int _smoothingWindowHalfSize=8;
  int _maxPeriodODF;

  std::vector<Real> _phase_1;
  std::vector<Real> _phase_2;
  std::vector<Real> _spectrum_1;


 public:
  OnsetDetectionGlobal() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_onsetDetections, "onsetDetections", "the frame-wise values of the detection function");

    _frameCutter = AlgorithmFactory::create("FrameCutter");
    _windowing = AlgorithmFactory::create("Windowing");
    _spectrum = AlgorithmFactory::create("Spectrum");
    _fft = AlgorithmFactory::create("FFT");
    _cartesian2polar = AlgorithmFactory::create("CartesianToPolar");
    _movingAverage = AlgorithmFactory::create("MovingAverage");
    _erbbands = AlgorithmFactory::create("ERBBands");
    _autocorrelation = AlgorithmFactory::create("AutoCorrelation");
  }

  ~OnsetDetectionGlobal() {
    if (_frameCutter) delete _frameCutter;
    if (_windowing) delete _windowing;
    if (_spectrum) delete _spectrum;
  }

  void declareParameters() {
    declareParameter("method", "the method used for onset detection", "{infogain,beat_emphasis}", "infogain");
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.0);
    declareParameter("frameSize", "the frame size for computing onset detection function", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size for computing onset detection function", "(0,inf)", 512);
  }

  void reset();
  void configure();
  void compute();

  void computeInfoGain();
  void computeBeatEmphasis();

  static const char* name;
  static const char* version;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmcomposite.h"
#include "pool.h"

namespace essentia {
namespace streaming {

class OnsetDetectionGlobal : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _signal;
  Source<Real> _onsetDetections;

  Pool _pool;
  Algorithm* _poolStorage;
  standard::Algorithm * _onsetDetectionGlobal;


 public:
  OnsetDetectionGlobal();
  ~OnsetDetectionGlobal();

  void declareParameters() {
    declareParameter("method", "the method used for onset detection", "{infogain,beat_emphasis}", "infogain");
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.0);
    declareParameter("frameSize", "the frame size for computing onset detection function", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size for computing onset detection function", "(0,inf)", 512);
  }

  void configure() {
    _onsetDetectionGlobal->configure(INHERIT("method"),
                                     INHERIT("sampleRate"),
                                     INHERIT("frameSize"),
                                     INHERIT("hopSize"));
  }

  void declareProcessOrder() {
    declareProcessStep(SingleShot(_poolStorage));
    declareProcessStep(SingleShot(this));
  }

  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ONSETDETECTIONGLOBAL_H
