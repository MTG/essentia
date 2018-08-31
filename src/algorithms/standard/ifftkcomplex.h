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

#ifndef ESSENTIA_IFFTKCOMPLEX_H
#define ESSENTIA_IFFTKCOMPLEX_H

#include "algorithm.h"
#include "threading.h"
#include <complex>
#include "kiss_fft130/kiss_fft.h"

namespace essentia {
namespace standard {

class IFFTKComplex : public Algorithm {

 protected:
  Input<std::vector<std::complex<Real> > > _fft;
  Output<std::vector<std::complex<Real> > > _signal;

 public:
  IFFTKComplex() : _input(0), _output(0), _fftCfg(0) {
    declareInput(_fft, "fft", "the input frame");
    declareOutput(_signal, "frame", "the IFFT of the input frame");
  }

  ~IFFTKComplex();

  void declareParameters() {
    declareParameter("size", "the expected size of the input frame. This is purely optional and only targeted at optimizing the creation time of the FFT object", "[1,inf)", 1024);
    declareParameter("normalize", "wheter to normalize the output by the FFT length.", "{true,false}", true);
  }


  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

 protected:
  kiss_fft_cfg _fftCfg;
  int _fftPlanSize;
  std::complex<Real>* _input;
  std::complex<Real>* _output;
  bool _normalize;
    
  void createFFTObject(int size);
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class IFFTKComplex : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::complex<Real> > > _fft;
  Source<std::vector<std::complex<Real> > > _signal;

 public:
  IFFTKComplex() {
    declareAlgorithm("IFFTC");
    declareInput(_fft, TOKEN, "fft");
    declareOutput(_signal, TOKEN, "frame");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_IFFTKCOMPLEX_H
