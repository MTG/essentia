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

#ifndef ESSENTIA_NSGICONSTANTQ_H
#define ESSENTIA_NSGICONSTANTQ_H

#include "algorithm.h"
#include "algorithmfactory.h"


namespace essentia {
namespace standard {

class NSGIConstantQ : public Algorithm {
 protected:
  Output<std::vector<Real> > _signal;
  Input<std::vector<std::vector<std::complex<Real> > > >_constantQ ;
  Input<std::vector<std::complex<Real> > > _constantQDC;
  Input<std::vector<std::complex<Real> > > _constantQNF;
  Input<std::vector<Real> > _shiftsOut;
  Input<std::vector<Real> > _winsLenOut;
  Input<std::vector<std::vector<Real> > > _freqWins;

 public:
  NSGIConstantQ() {
    declareOutput(_signal, "frame", "the input frame (vector)");
    declareInput(_constantQ, "constantq", "the Non Stationary Gabor transform based constant Q transform of the input frame");
    declareInput(_constantQDC, "constantqdc", "the DC component of the constant Q transform. Needed for the time reconstruction");
    declareInput(_constantQNF, "constantqnf", "the Nyquist Frequency component of the constant Q transform. Needed for the time reconstruction");
    declareInput(_shiftsOut, "windowShifts", "Amount of bins from the center of each frequency window to the base band. Needed for the time reconstruction");
    declareInput(_winsLenOut, "windowLenghts", "Longitudes of the frequency windows used in the transform. Needed for the time reconstruction");
    declareInput(_freqWins, "frequencyWindows", "Frequency windows used in the transform. Needed for the time reconstruction");


    _fft = AlgorithmFactory::create("FFTC");
    _ifft = AlgorithmFactory::create("IFFTC");
  }

  ~NSGIConstantQ() {
    if (_fft) delete _fft;
    if (_ifft) delete _ifft;
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the input", "(0,inf)", 1024);
    declareParameter("minFrequency", "the minimum frequency", "(0,inf)", 27.5);
    declareParameter("maxFrequency", "the maximum frequency", "(0,inf)", 55);
    declareParameter("binsPerOctave", "the number of bins per octave", "[1,inf)", 12);
    declareParameter("sampleRate", "the desired sampling rate [Hz]", "[0,inf)", 44100.);
    declareParameter("rasterize", "hop sizes for each frequency channel. With 'none' each frequency channel is distinct. 'full' sets the hop sizes of all the channels to the smallest. 'piecewise' rounds down the hop size to a power of two", "{none,full,piecewise}", "full");
    declareParameter("phaseMode", "'local' to use zero-centered filters. 'global' to use a mapping function [2]", "{local,global}", "global");
    declareParameter("gamma", "The bandwidth of each filter is given by Bk = 1/Q * fk + gamma", "[0,inf)", 0);
    declareParameter("normalize", "coefficient normalization", "{sine,impulse,none}", "sine");
    declareParameter("window","the type of window for the frequency filter. See 'Windowing'","{hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}","hann");
    declareParameter("minimumWindow", "Minimum size allowed for the windows", "[2,inf)", 4);
    declareParameter("windowSizeFactor", "Window sizes are rounded to multiples of this", "[1,inf)", 1);
    declareParameter("INSQConstantQdata", "Flag to output the data needed for the reconstruction", "{true,false}", true);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

 protected:

  Algorithm* _ifft;
  Algorithm* _fft;

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
  int _minimumWindow;
  int _windowSizeFactor;
  bool _INSQConstantQdata;

  //windowing vectors
  //std::vector< std::vector<Real> > _freqWins;
  std::vector<int> _shifts;
  std::vector<Real> _shiftsReal;
  std::vector<Real> _shiftsFreq;
  std::vector<int> _winsLen;
  std::vector<Real> _winsLenReal;
  std::vector<Real> _baseFreqs;
  int _binsNum;

  std::vector<int> _posit;
  std::vector< std::vector<std::complex<Real> > > _fr;


};

}
}

#endif // ESSENTIA_NSGICONSTANTQ_H
