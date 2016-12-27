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
  Input<std::vector<std::complex<Real> > > _signal;
  Output<std::vector<std::complex<Real> > > _constantQ;

  Algorithm* _fft;

  std::vector<double> _CQdata;
  
  double _sampleRate; //unsigned int _FS;
  double _minFrequency;
  double _maxFrequency;
  double _dQ; // Work out Q value for Filter bank
  double _threshold; // ConstantQ threshold for kernel generation
  unsigned int _numWin;
  unsigned int _hop;
  unsigned int _binsPerOctave;  
  unsigned int _FFTLength;
  unsigned int _uK; // Number of constant Q bins

  struct SparseKernel {
    std::vector<double> _sparseKernelReal;
    std::vector<double> _sparseKernelImag;
    std::vector<unsigned> _sparseKernelIs; 
    std::vector<unsigned> _sparseKernelJs;
  };

  SparseKernel *m_sparseKernel;

  double hamming(int len, int n) {
    return 0.54 - 0.46*cos(2 * M_PI * n / len);
  }

 public:
  ConstantQ() {
    declareInput(_signal, "frame", "the input frame (complex)");
    declareOutput(_constantQ, "constantq", "the Constant Q transform of the input frame");

    _fft = AlgorithmFactory::create("FFTC"); //FFT with complex input
  }

  ~ConstantQ() {
    delete _fft;
    if (m_sparseKernel) delete m_sparseKernel;
  }

  int sizeFFT() { return _FFTLength; }

  void declareParameters() {
    declareParameter("minFrequency", "the minimum frequency", "[1,inf)", 55.);
    declareParameter("maxFrequency", "the maximum frequency", "[1,inf)", 7040.);
    declareParameter("binsPerOctave", "the number of bins per octave", "[1,inf)", 24);    
    declareParameter("sampleRate", "the desired sampling rate [Hz]", "[0,inf)", 44100.);  
    declareParameter("threshold", "threshold value", "[0,inf)", 0.0005);       
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
  Sink<std::vector<std::complex<Real> > > _signal;
  Source<std::vector<std::complex<Real> > > _constantQ;

 public:
  ConstantQ() {
    declareAlgorithm("ConstantQ");
    declareInput(_signal, TOKEN, "frame");
    declareOutput(_constantQ, TOKEN, "constantq");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CONSTANTQ_H
