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

#ifndef ESSENTIA_PITCHSALIENCEFUNCTION_H
#define ESSENTIA_PITCHSALIENCEFUNCTION_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class PitchSalienceFunction : public Algorithm {

 private:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Output<std::vector<Real> > _salienceFunction;

  Real _binResolution;
  Real _referenceFrequency;
  Real _magnitudeThreshold;
  Real _magnitudeCompression;
  int _numberHarmonics;
  Real _harmonicWeight;


  std::vector<Real> _harmonicWeights;     // precomputed vector of weights for n-th harmonics
  std::vector<Real> _nearestBinsWeights;  // precomputed vector of weights for salience propagation to nearest bins
  int _numberBins;
  int _binsInSemitone;                // number of bins in a semitone
  Real _binsInOctave;                 // number of bins in an octave
  Real _referenceTerm;                // precomputed addition term used for Hz to cent bin conversion
  Real _magnitudeThresholdLinear;     // fraction of maximum magnitude in frame corresponding to _magnitudeCompression difference in dBs

  int frequencyToCentBin(Real frequency);

 public:
  PitchSalienceFunction() {
    declareInput(_frequencies, "frequencies", "the frequencies of the spectral peaks [Hz]");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the spectral peaks");
    declareOutput(_salienceFunction, "salienceFunction", "array of the quantized pitch salience values");
  }

  ~PitchSalienceFunction() {
  };

  void declareParameters() {
    declareParameter("binResolution", "salience function bin resolution [cents]", "(0,inf)", 10.0);
    declareParameter("referenceFrequency", "the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin", "(0,inf)", 55.0);
    declareParameter("magnitudeThreshold", "peak magnitude threshold (maximum allowed difference from the highest peak in dBs)", "[0,inf)",  40.0);
    declareParameter("magnitudeCompression", "magnitude compression parameter (=0 for maximum compression, =1 for no compression)", "(0,1]", 1.0);
    declareParameter("numberHarmonics", "number of considered harmonics", "[1,inf)", 20);
    declareParameter("harmonicWeight", "harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay)", "(0,1)", 0.8);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* version;
  static const char* description;

}; // class PitchSalienceFunction

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PitchSalienceFunction : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<std::vector<Real> > _salienceFunction;

 public:
  PitchSalienceFunction() {
    declareAlgorithm("PitchSalienceFunction");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_salienceFunction, TOKEN, "salienceFunction");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PITCHSALIENCEFUNCTION_H
