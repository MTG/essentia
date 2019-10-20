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

#ifndef ESSENTIA_WELCH_H
#define ESSENTIA_WELCH_H

#include "essentiamath.h"
#include "algorithm.h"
#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class Welch : public Algorithm {
 protected:
  Input<std::vector<Real> > _frame;
  Output<std::vector<Real> > _psd;
  Real _sampleRate;
  uint _frameSize;
  uint _fftSize;
  uint _padding;
  uint _spectSize;
  uint _averagingFrames;
  std::string _scaling;
  std::string _windowType;

  Real _normalization;

  Algorithm* _window;
  Algorithm* _powerSpectrum;

  std::vector<std::vector<Real> > _psdBuffer;
  std::vector<Real> _windowed;
  std::vector<Real> _powerSpectrumFrame;

 public:
  Welch() {
    declareInput(_frame, "frame", "the input stereo audio signal");
    declareOutput(_psd, "psd", "Power Spectral Density [dB] or [dB/Hz]");

    _window = AlgorithmFactory::create("Windowing");
    _powerSpectrum  = AlgorithmFactory::create("PowerSpectrum");
  }

  ~Welch() {
    if (_window) delete _window;
    if (_powerSpectrum) delete _powerSpectrum;
  };

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the expected size of the input audio signal (this is an optional parameter to optimize memory allocation)", "(0,inf)", 512);
    declareParameter("windowType", "the window type, which can be 'hamming', 'hann', 'triangular', 'square' or 'blackmanharrisXX'", "{hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}", "hann");
    declareParameter("fftSize", "size of the FFT. Zero padding is added if this is larger the input frame size.", "(0,inf)", 1024);  
    declareParameter("scaling", "'density' normalizes the result to the bandwidth while 'power' outputs the unnormalized power spectrum", "{density,power}", "density");  
    declareParameter("averagingFrames", "amount of frames to average", "(0,inf)", 10);  
  };

  void configure();
  void compute();
  void reset();
  void initBuffers();

  static const char* name;
  static const char* category;
  static const char* description;
};


} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Welch : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frame;
  Source<std::vector<Real> > _psd;

 public:
  Welch() {
    declareAlgorithm("Welch");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_psd, TOKEN, "psd");
  }
};

} // namespace streaming
} // namespace essentia



#endif // ESSENTIA_WELCH_H
