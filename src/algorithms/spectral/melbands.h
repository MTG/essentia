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

#ifndef ESSENTIA_MELBANDS_H
#define ESSENTIA_MELBANDS_H

#include "essentiamath.h"
#include "algorithm.h"
#include "algorithmfactory.h"


namespace essentia {
namespace standard {

class MelBands : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrumInput;
  Output<std::vector<Real> > _bandsOutput;

  Algorithm* _triangularBands;

 public:
  MelBands() {
    declareInput(_spectrumInput, "spectrum", "the audio spectrum");
    declareOutput(_bandsOutput, "bands", "the energy in mel bands");

    _triangularBands = AlgorithmFactory::create("TriangularBands");
  }

  ~MelBands() {
    if (_triangularBands) delete _triangularBands;
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the spectrum", "(1,inf)", 1025);
    declareParameter("numberBands", "the number of output bands", "(1,inf)", 24);
    declareParameter("sampleRate", "the sample rate", "(0,inf)", 44100.);
    declareParameter("lowFrequencyBound", "a lower-bound limit for the frequencies to be included in the bands", "[0,inf)", 0.0);
    declareParameter("highFrequencyBound", "an upper-bound limit for the frequencies to be included in the bands", "[0,inf)", 22050.0);
    declareParameter("warpingFormula", "The scale implementation type: 'htkMel' scale from the HTK toolkit [2, 3] (default) or 'slaneyMel' scale from the Auditory toolbox [4]","{slaneyMel,htkMel}","htkMel");
    declareParameter("weighting", "type of weighting function for determining triangle area","{warping,linear}","warping");
    declareParameter("normalize", "spectrum bin weights to use for each mel band: 'unit_max' to make each mel band vertex equal to 1, 'unit_sum' to make each mel band area equal to 1 summing the actual weights of spectrum bins, 'unit_area' to make each triangle mel band area equal to 1 normalizing the weights of each triangle by its bandwidth","{unit_sum,unit_tri,unit_max}", "unit_sum");
    declareParameter("type", "'power' to output squared units, 'magnitude' to keep it as the input","{magnitude,power}", "power");
    declareParameter("log", "compute log-energies (log10 (1 + energy))","{true,false}", false);

  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

 protected:

  void calculateFilterFrequencies();
  void setWarpingFunctions(std::string warping, std::string weighting);

  std::vector<std::vector<Real> > _filterCoefficients;
  std::vector<Real> _filterFrequencies;
  int _numBands;
  Real _sampleRate;

  std::string _normalization;
  std::string _type;
  std::string _weighting;
  typedef Real (*funcPointer)(Real);

  funcPointer _inverseWarper;
  funcPointer _warper;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class MelBands : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrumInput;
  Source<std::vector<Real> > _bandsOutput;

 public:
  MelBands() {
    declareAlgorithm("MelBands");
    declareInput(_spectrumInput, TOKEN, "spectrum");
    declareOutput(_bandsOutput, TOKEN, "bands");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_MELBANDS_H
