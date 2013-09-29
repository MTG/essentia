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

#ifndef ESSENTIA_ERBBANDS_H
#define ESSENTIA_ERBBANDS_H

#include "essentiamath.h"
#include "algorithm.h"
#include <complex>

namespace essentia {
namespace standard {

class ERBBands : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrumInput;
  Output<std::vector<Real> > _bandsOutput;

 public:
  ERBBands() {
    declareInput(_spectrumInput, "spectrum", "the audio spectrum");
    declareOutput(_bandsOutput, "bands", "the energies/magnitudes of each band");
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the spectrum", "(1,inf)", 1025);
    declareParameter("numberBands", "the number of output bands", "(1,inf)", 40);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("lowFrequencyBound", "a lower-bound limit for the frequencies to be included in the bands", "[0,inf)", 50.0);
    declareParameter("highFrequencyBound", "an upper-bound limit for the frequencies to be included in the bands", "[0,inf)", 22050.0);
    declareParameter("width", "filter width with respect to ERB", "(0,inf)", 1.0);
    declareParameter("type", "compute energies or magnitudes", "{energy,magnitude}", "energy");
  }

  void configure();
  void compute();

  static const char* name;
  static const char* version;
  static const char* description;

 protected:

  void createFilters(int spectrumSize);
  void calculateFilterFrequencies();

  std::vector<std::vector<Real> > _filterCoefficients;
  std::vector<Real> _filterFrequencies;
  int _numberBands;

  Real _sampleRate;
  Real _maxFrequency;
  Real _minFrequency;
  Real _width;
  std::string _type;

  static const Real EarQ;
  static const Real minBW;

};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {


class ERBBands : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrumInput;
  Source<std::vector<Real> > _bandsOutput;

 public:
  ERBBands() {
    declareAlgorithm("ERBBands");
    declareInput(_spectrumInput, TOKEN, "spectrum");
    declareOutput(_bandsOutput, TOKEN, "bands");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_ERBBands_H
