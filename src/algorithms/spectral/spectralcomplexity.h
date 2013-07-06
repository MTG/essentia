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

#ifndef ESSENTIA_SPECTRALCOMPLEXITY_H
#define ESSENTIA_SPECTRALCOMPLEXITY_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class SpectralComplexity : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _spectralComplexity;

  Algorithm* _spectralPeaks;

 public:
  SpectralComplexity() {
    declareInput(_spectrum, "spectrum", "the input spectrum");
    declareOutput(_spectralComplexity, "spectralComplexity", "the spectral complexity of the input spectrum");

    _spectralPeaks = AlgorithmFactory::create("SpectralPeaks");
  }

  ~SpectralComplexity() {
    delete _spectralPeaks;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("magnitudeThreshold", "the minimum spectral-peak magnitude that contributes to spectral complexity", "[0,inf)", 0.005);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class SpectralComplexity : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _spectralComplexity;

 public:
  SpectralComplexity() {
    declareAlgorithm("SpectralComplexity");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_spectralComplexity, TOKEN, "spectralComplexity");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SPECTRALCOMPLEXITY_H
