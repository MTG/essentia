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
#include "essentiamath.h"
#include <complex>
#include <vector>


namespace essentia {
namespace standard {

class ConstantQ : public Algorithm {
 protected:
  Input<std::vector<Real> > _frame;
  Output<std::vector<std::complex<Real> > > _constantQ;

  Algorithm* _fftc;
  Algorithm* _windowing;
  Algorithm* _fft;

  std::vector<double> _CQdata;
  std::vector<std::complex<Real> > _fftData;
  
  double _sampleRate;
  double _minFrequency;
  double _maxFrequency;
  double _Q;            // constant Q factor
  double _threshold;    // threshold for kernel generation
  double _scale;

  unsigned int _numWin;
  unsigned int _binsPerOctave;
  unsigned int _windowSize;
  unsigned int _inputFFTSize;
  unsigned int _numberBins;
  unsigned int _minimumKernelSize;

  bool _zeroPhase;

  struct SparseKernel {
    std::vector<double> real;
    std::vector<double> imag;
    std::vector<unsigned> i; 
    std::vector<unsigned> j;
  };

  SparseKernel _sparseKernel;


 public:
  ConstantQ() {
    declareInput(_frame, "frame", "the windowed input audio frame");
    declareOutput(_constantQ, "constantq", "the Constant Q transform");

    _fftc = AlgorithmFactory::create("FFTC"); //FFT with complex input
    _windowing = AlgorithmFactory::create("Windowing", "zeroPhase", false);
    _fft = AlgorithmFactory::create("FFT");
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
    declareParameter("threshold", "bins whose magnitude is below this quantile are discarded", "[0,1)", 0.01);
    declareParameter("scale", "filters scale. Larger values use longer windows", "[0,inf)", 1.0);
    declareParameter("windowType", "the window type, which can be 'hamming', 'hann', 'triangular', 'square' or 'blackmanharrisXX'", "{hamming,hann,hannnsgcq,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}", "hann");
    declareParameter("minimumKernelSize", "minimum size allowed for frequency kernels", "[2,inf)", 4);
    declareParameter("zeroPhase", "a boolean value that enables zero-phase windowing. Input audio frames should be windowed with the same phase mode", "{true,false}", true);

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
  Sink<std::vector<Real> > _frame;
  Source<std::vector<std::complex<Real> > > _constantQ;

 public:
  ConstantQ() {
    declareAlgorithm("ConstantQ");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_constantQ, TOKEN, "constantq");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CONSTANTQ_H
