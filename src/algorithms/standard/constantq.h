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

#ifndef ESSENTIA_CONSTANTQ_H
#define ESSENTIA_CONSTANTQ_H

#include "algorithm.h"
#include "algorithmfactory.h"
#include <complex>
#include <vector>


namespace essentia {
namespace standard {

class ConstantQ : public Algorithm {
 protected:
  Input<std::vector<std::complex<Real> > > _fft;
  Output<std::vector<std::complex<Real> > > _constantQ;

  Algorithm* _fftc;
  Algorithm* _windowing;

  std::vector<double> _CQdata;
  
  double _sampleRate;
  double _minFrequency;
  double _maxFrequency;
  double _Q;            // constant Q factor
  double _threshold;    // threshold for kernel generation
  unsigned int _numWin;
  unsigned int _binsPerOctave;  
  unsigned int _windowSize;
  unsigned int _inputFFTSize;
  unsigned int _numberBins;

  struct SparseKernel {
    std::vector<double> real;
    std::vector<double> imag;
    std::vector<unsigned> i; 
    std::vector<unsigned> j;
  };

  struct SparseKernel _sparseKernel;


 public:
  ConstantQ() {
    declareInput(_fft, "fft", "the input FFT frame (complex, non-negative part)");
    declareOutput(_constantQ, "constantq", "the Constant Q transform");

    _fftc = AlgorithmFactory::create("FFTC"); //FFT with complex input
    _windowing = AlgorithmFactory::create("Windowing");
  }

  ~ConstantQ() {
    delete _fftc;
    delete _windowing;
  }

  void declareParameters() {
    declareParameter("minFrequency", "minimum frequency [Hz]", "[1,inf)", 32.7);
    declareParameter("numberBins", "number of frequency bins, starting at minFrequency", "[1,inf)", 84);
    declareParameter("binsPerOctave", "number of bins per octave", "[1,inf)", 12);    
    declareParameter("sampleRate", "FFT sampling rate [Hz]", "[0,inf)", 44100.);  
    declareParameter("threshold", "threshold value", "[0,inf)", 0.0005);
    // TODO: explain threshold better 
  }

  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class ConstantQ : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::complex<Real> > > _fft;
  Source<std::vector<std::complex<Real> > > _constantQ;

 public:
  ConstantQ() {
    declareAlgorithm("ConstantQ");
    declareInput(_fft, TOKEN, "fft");
    declareOutput(_constantQ, TOKEN, "constantq");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CONSTANTQ_H
