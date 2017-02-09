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
  Output<std::vector< std::vector<Real> > >_constantQ ;

 public:
  NSGConstantQ() {
    declareInput(_signal, "frame", "the input frame (vector)");
    declareOutput(_constantQ, "constantq", "the Non Stationary Gabor transform bAsed constant Q transform of the input frame");

    _fft = AlgorithmFactory::create("FFT"); //FFT with complex input
    _ifft = AlgorithmFactory::create("IFFT");
    _windowing = AlgorithmFactory::create("Windowing");
  }

  ~NSGConstantQ() {
    delete _fft;
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the input", "(0,inf)", 1024);
    declareParameter("minFrequency", "the minimum frequency", "(0,inf)", 27.5);
    declareParameter("maxFrequency", "the maximum frequency", "(0,inf)", 55);
    declareParameter("binsPerOctave", "the number of bins per octave", "[1,inf)", 12);
    declareParameter("sampleRate", "the desired sampling rate [Hz]", "[0,inf)", 44100.);
    // declareParameter("threshold", "threshold value", "[0,inf)", 0.0005);
    declareParameter("rasterize", "hop sizes for each frequency channel. With 'none' each frequency channel is distinct. 'full' sets the hop sizes of all the channels to the smallest. 'piecewise' rounds down the hop size to a power of two", "{none,full,piecewise}", "full");
    declareParameter("phaseMode", "'local' to use zero-centered filters. 'global' to use a mapping function [2]", "{local,global}", "global");
    declareParameter("gamma", "The bandwidth of each filter is given by Bk = 1/Q * fk + gamma", "[0,inf)", 0);
    declareParameter("normalize", "coefficient normalization", "{sinusoid,impulse,none}", "sinusoid");
    declareParameter("window","the type of window for the frequency filter. See 'Windowing'","{hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}","hann");
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

  //Variables for the input parameters
  Real _minFrequency;
  Real _maxFrequency;
  Real _sr;
  Real _binsPerOctave;
  int _inputSize;
  Real _gamma;
  std::string _rasterize;
  std::string _phaseMode;
  std::string _normalize;


  //windowing vectors
  std::vector< std::vector<Real> > _freqFilters;
  std::vector<int > _shifts;
  std::vector<Real> _shiftsFreq;
  std::vector<int> _filtersLength;
  std::vector<Real> _baseFreqs;

  int _binsNum;
};

}
}

#endif // ESSENTIA_NSGCONSTANTQ_H
