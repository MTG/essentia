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

#ifndef ESSENTIA_MFCC_H
#define ESSENTIA_MFCC_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class MFCC : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrum;
  Output<std::vector<Real> > _bands;
  Output<std::vector<Real> > _mfcc;

  Algorithm* _melFilter;
  Algorithm* _dct;

  std::vector<Real> _logbands;

 public:
  MFCC() {
    declareInput(_spectrum, "spectrum", "the audio spectrum");
    declareOutput(_bands, "bands" , "the energies in mel bands");
    declareOutput(_mfcc, "mfcc", "the mel frequency cepstrum coefficients");

    _melFilter = AlgorithmFactory::create("MelBands");
    _dct = AlgorithmFactory::create("DCT");
  }

  ~MFCC() {
    if (_melFilter) delete _melFilter;
    if (_dct) delete _dct;
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of input spectrum", "(1,inf)", 1025);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("numberBands", "the number of mel-bands in the filter", "[1,inf)", 40);
    declareParameter("numberCoefficients", "the number of output mel coefficients", "[1,inf)", 13);
    declareParameter("lowFrequencyBound", "the lower bound of the frequency range [Hz]", "[0,inf)", 0.);
    declareParameter("highFrequencyBound", "the upper bound of the frequency range [Hz]", "(0,inf)", 11000.);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} //namespace standard
} //namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class MFCC : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<std::vector<Real> > _bands;
  Source<std::vector<Real> > _mfcc;

 public:
  MFCC() {
    declareAlgorithm("MFCC");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_bands, TOKEN, "bands");
    declareOutput(_mfcc, TOKEN, "mfcc");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_MFCC_H
