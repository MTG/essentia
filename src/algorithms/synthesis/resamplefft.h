/*
 * Copyright (C) 2006-2015  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_ResampleFFT_H
#define ESSENTIA_ResampleFFT_H


#include "algorithm.h"
#include "algorithmfactory.h"

#include <fstream>

namespace essentia {
namespace standard {

class ResampleFFT : public Algorithm {

 protected:
  Input<std::vector<Real> > _input;
  Output<std::vector<Real> > _output;

  // for resample function
  Algorithm* _fft;
  Algorithm* _ifft;

 public:
  ResampleFFT() {
    declareInput(_input, "input", "input array");
    declareOutput(_output, "output", "output resample array");

    // for resample
    _fft = AlgorithmFactory::create("FFT");
    _ifft = AlgorithmFactory::create("IFFT");

  }

  ~ResampleFFT() {

    delete _fft;
    delete _ifft;

  }

  void declareParameters() {
    declareParameter("inSize", "the size of the input sequence. It needss to be even-sized.", "[1,inf)", 128);
    declareParameter("outSize", "the size of the output sequence. It needss to be even-sized.", "[1,inf)", 128);
  }

  void configure();
  void compute();


  static const char* name;
  static const char* category;
  static const char* description;


};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class ResampleFFT : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _input;
  Source<std::vector<Real> > _output;

 public:
  ResampleFFT() {
    declareAlgorithm("ResampleFFT");
    declareInput(_input, TOKEN, "input");
    declareOutput(_output, TOKEN, "output");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_ResampleFFT_H
