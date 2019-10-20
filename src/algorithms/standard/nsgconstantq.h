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

#ifndef ESSENTIA_NSGCONSTANTQ_H
#define ESSENTIA_NSGCONSTANTQ_H

#include "algorithm.h"
#include "algorithmfactory.h"


namespace essentia {
namespace standard {

class NSGConstantQ : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector< std::vector<std::complex<Real> > > >_constantQ ;
  Output<std::vector<std::complex<Real> > > _constantQDC;
  Output<std::vector<std::complex<Real> > > _constantQNF;

 public:
  NSGConstantQ() {
    declareInput(_signal, "frame", "the input frame (vector)");
    declareOutput(_constantQ, "constantq", "the constant Q transform of the input frame");
    declareOutput(_constantQDC, "constantqdc", "the DC band transform of the input frame. Only needed for the inverse transform");
    declareOutput(_constantQNF, "constantqnf", "the Nyquist band transform of the input frame. Only needed for the inverse transform");

    _fft = AlgorithmFactory::create("FFT");
    _ifft = AlgorithmFactory::create("IFFTC");
    _windowing = AlgorithmFactory::create("Windowing");
  }

  ~NSGConstantQ() {
    if (_fft) delete _fft;
    if (_ifft) delete _ifft;
    if (_windowing) delete _windowing;
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the input", "(0,inf)", 4096);
    declareParameter("minFrequency", "the minimum frequency", "(0,inf)", 27.5);
    declareParameter("maxFrequency", "the maximum frequency", "(0,inf)", 7040.);
    declareParameter("binsPerOctave", "the number of bins per octave", "[1,inf)", 48);
    declareParameter("sampleRate", "the desired sampling rate [Hz]", "[0,inf)", 44100.);
    declareParameter("rasterize", "hop sizes for each frequency channel. With 'none' each frequency channel is distinct. 'full' sets the hop sizes of all the channels to the smallest. 'piecewise' rounds down the hop size to a power of two", "{none,full,piecewise}", "full");
    declareParameter("phaseMode", "'local' to use zero-centered filters. 'global' to use a phase mapping function as described in [1]", "{local,global}", "global");
    declareParameter("gamma", "The bandwidth of each filter is given by Bk = 1/Q * fk + gamma", "[0,inf)", 0);
    declareParameter("normalize", "coefficient normalization", "{sine,impulse,none}", "none");
    declareParameter("window","the type of window for the frequency filters. It is not recommended to change the default window.","{hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}","hannnsgcq");
    declareParameter("minimumWindow", "minimum size allowed for the windows", "[2,inf)", 4);
    declareParameter("windowSizeFactor", "window sizes are rounded to multiples of this", "[1,inf)", 1);
  }

  void compute();
  void configure();
  void designWindow();
  void createCoefficients();
  void normalize();

  static const char* name;
  static const char* category;
  static const char* description;

 protected:

  Algorithm* _ifft;
  Algorithm* _fft;
  Algorithm* _windowing;

  // variables for the input parameters
  Real _minFrequency;
  Real _maxFrequency;
  Real _sr;
  Real _binsPerOctave;
  int _inputSize;
  Real _gamma;
  std::string _rasterize;
  std::string _phaseMode;
  std::string _normalize;
  int _minimumWindow;
  int _windowSizeFactor;

  // windowing vectors
  std::vector< std::vector<Real> > _freqWins;
  std::vector<int> _shifts;
  std::vector<int> _winsLen;
  std::vector<Real> _baseFreqs;
  int _binsNum;
};

}
}


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class NSGConstantQ : public StreamingAlgorithmWrapper {
 protected:
  Sink<std::vector<Real> > _signal;

  Source<std::vector<std::vector<std::complex<Real> > > >_constantQ ;
  Source<std::vector<std::complex<Real> > > _constantQDC;
  Source<std::vector<std::complex<Real> > > _constantQNF;

 public:
  NSGConstantQ() {
    declareAlgorithm("NSGConstantQ");
    declareInput(_signal, TOKEN, "frame");
    declareOutput(_constantQ, TOKEN, "constantq");
    declareOutput(_constantQDC, TOKEN, "constantqdc");
    declareOutput(_constantQNF, TOKEN, "constantqnf");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_NSGCONSTANTQ_H
