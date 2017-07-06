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

#ifndef ESSENTIA_BFCC_H
#define ESSENTIA_BFCC_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class BFCC : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrum;
  Output<std::vector<Real> > _bands;
  Output<std::vector<Real> > _bfcc;

  Algorithm* _triangularBarkFilter;
  Algorithm* _dct;

  std::vector<Real> _logbands;

  typedef  Real (*funcPointer)(Real);
  funcPointer _compressor;

  void setCompressor(std::string logType);

 public:
  BFCC() {
    declareInput(_spectrum, "spectrum", "the audio spectrum");
    declareOutput(_bands, "bands" , "the energies in bark bands");
    declareOutput(_bfcc, "bfcc", "the bark frequency cepstrum coefficients");

    _triangularBarkFilter = AlgorithmFactory::create("TriangularBarkBands");
    _dct = AlgorithmFactory::create("DCT");
  }

  ~BFCC() {
    if (_triangularBarkFilter) delete _triangularBarkFilter;
    if (_dct) delete _dct;
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of input spectrum", "(1,inf)", 1025);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("numberBands", "the number of bark bands in the filter", "[1,inf)", 40);
    declareParameter("numberCoefficients", "the number of output cepstrum coefficients", "[1,inf)", 13);
    declareParameter("lowFrequencyBound", "the lower bound of the frequency range [Hz]", "[0,inf)", 0.);
    declareParameter("highFrequencyBound", "the upper bound of the frequency range [Hz]", "(0,inf)", 11000.);    
    declareParameter("weighting", "type of weighting function for determining triangle area","{warping,linear}","warping");
    declareParameter("normalize", "'unit_max' makes the vertex of all the triangles equal to 1, 'unit_sum' makes the area of all the triangles equal to 1","{unit_sum,unit_max}", "unit_sum");
    declareParameter("type", "use magnitude or power spectrum","{magnitude,power}", "power");
    declareParameter("dctType", "the DCT type", "[2,3]", 2);
    declareParameter("liftering", "the liftering coefficient. Use '0' to bypass it", "[0,inf)", 0);
    declareParameter("logType","logarithmic compression type. Use 'dbpow' if working with power and 'dbamp' if working with magnitudes","{natural,dbpow,dbamp,log}","dbamp");

  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} //namespace standard
} //namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class BFCC : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<std::vector<Real> > _bands;
  Source<std::vector<Real> > _bfcc;

 public:
  BFCC() {
    declareAlgorithm("BFCC");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_bands, TOKEN, "bands");
    declareOutput(_bfcc, TOKEN, "bfcc");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BFCC_H
