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
  std::string _logType;
  Real _silenceThreshold;
  Real _dbSilenceThreshold;
  Real _logSilenceThreshold;

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
    declareParameter("warpingFormula", "The scale implementation type: 'htkMel' scale from the HTK toolkit [2, 3] (default) or 'slaneyMel' scale from the Auditory toolbox [4]","{slaneyMel,htkMel}","htkMel");    declareParameter("weighting", "type of weighting function for determining triangle area","{warping,linear}","warping");
    declareParameter("normalize", "spectrum bin weights to use for each mel band: 'unit_max' to make each mel band vertex equal to 1, 'unit_sum' to make each mel band area equal to 1 summing the actual weights of spectrum bins, 'unit_area' to make each triangle mel band area equal to 1 normalizing the weights of each triangle by its bandwidth","{unit_sum,unit_tri,unit_max}", "unit_sum");
    declareParameter("type", "use magnitude or power spectrum","{magnitude,power}", "power");
    declareParameter("silenceThreshold", "silence threshold for computing log-energy bands", "(0,inf)", 1e-10);
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
