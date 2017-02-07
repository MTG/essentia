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

#ifndef ESSENTIA_SPECTRUMTOCENT_H
#define ESSENTIA_SPECTRUMTOCENT_H

#include "algorithm.h"
#include "essentiautil.h"
#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace standard {

class SpectrumToCent : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrumInput;
  Output<std::vector<Real> > _bandsOutput;
  Output<std::vector<Real> > _freqOutput;

  std::vector<Real> _bandFrequencies;
  std::vector<Real> _freqBands;

  int _nBands;
  Real _centBinRes;
  Real _minFrequency;
  Real _sampleRate;

  Algorithm* _triangularBands;

  void calculateFilterFrequencies();

 public:
  SpectrumToCent() {
    declareInput(_spectrumInput, "spectrum", "the input spectrum (must be greater than size one)");
    declareOutput(_bandsOutput, "bands", "the energy in each band");
    declareOutput(_freqOutput, "frequencies", "the central frequency of each band");
    _triangularBands = AlgorithmFactory::create("TriangularBands");
  }

  void declareParameters() {
    declareParameter("inputSize", "the size of the spectrum", "(1,inf)", 32768);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("log", "compute log-energies (log10 (1 + energy))","{true,false}", true);
    declareParameter("minimumFrequency","central frequency of the first band of the bank [Hz]", "(0, inf)", 164.);
    declareParameter("centBinResolution", "Width of each band in cents. Default is 10 cents","(0,inf)", 10.);
    declareParameter("bands", "number of bins to compute. Default is 720 (6 octaves with the default 'centBinResolution')","[1,inf)", 720);
    declareParameter("normalize", "use unit area or vertex equal to 1 triangles.","{unit_sum,unit_max}", "unit_sum");
    declareParameter("type", "use magnitude or power spectrum","{magnitude,power}", "power");
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

class SpectrumToCent : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrumInput;
  Source<std::vector<Real> > _bandsOutput;
  Source<std::vector<Real> > _freqOutput;

 public:
  SpectrumToCent() {
    declareAlgorithm("SpectrumToCent");
    declareInput(_spectrumInput, TOKEN, "spectrum");
    declareOutput(_bandsOutput, TOKEN, "bands");
    declareOutput(_freqOutput, TOKEN, "frequencies");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SPECTRUMTOCENT_H
