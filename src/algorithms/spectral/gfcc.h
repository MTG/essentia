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

#ifndef ESSENTIA_GFCC_H
#define ESSENTIA_GFCC_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class GFCC : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrum;
  Output<std::vector<Real> > _bands;
  Output<std::vector<Real> > _gfcc;

  Algorithm* _gtFilter;
  Algorithm* _dct;

  std::vector<Real> _logbands;

 public:
  GFCC() {
    declareInput(_spectrum, "spectrum", "the audio spectrum");
    declareOutput(_bands, "bands" , "the energies in ERB bands");
    declareOutput(_gfcc, "gfcc", "the gammatone feature cepstrum coefficients");

    _gtFilter = AlgorithmFactory::create("ERBBands");
    _dct = AlgorithmFactory::create("DCT");
  }

  ~GFCC() {
    if (_gtFilter) delete _gtFilter;
    if (_dct) delete _dct;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("numberBands", "the number of bands in the filter", "[1,inf)", 40);
    declareParameter("numberCoefficients", "the number of output cepstrum coefficients", "[1,inf)", 13);
    declareParameter("lowFrequencyBound", "the lower bound of the frequency range [Hz]", "[0,inf)", 40.);
    declareParameter("highFrequencyBound", "the upper bound of the frequency range [Hz]", "(0,inf)", 22050.);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* version;
  static const char* description;

};

} //namespace standard
} //namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class GFCC : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<std::vector<Real> > _bands;
  Source<std::vector<Real> > _gfcc;

 public:
  GFCC() {
    declareAlgorithm("GFCC");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_bands, TOKEN, "bands");
    declareOutput(_gfcc, TOKEN, "gfcc");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_GFCC_H
