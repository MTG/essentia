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

#ifndef ESSENTIA_SPECTRALCONTRAST_H
#define ESSENTIA_SPECTRALCONTRAST_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class SpectralContrast : public Algorithm {

protected:
  Input<std::vector<Real> >   _spectrum;
  Output<std::vector<Real> >  _spectralcontrast;
  Output<std::vector<Real> >  _valleys;
  std::vector<int>            _numberOfBinsInBands;
  Real                        _neighbourRatio;
  int                         _startAtBin;
  int                         _frameSize;

public:
  SpectralContrast() {
    declareInput(_spectrum, "spectrum", "the audio spectrum");
    declareOutput(_spectralcontrast, "spectralContrast", "the spectral contrast coefficients");
    declareOutput(_valleys, "spectralValley", "the magnitudes of the valleys");
  }

  ~SpectralContrast() {}

  void declareParameters() {
    declareParameter("frameSize", "the size of the fft frames", "[2,inf)", 2048);
    declareParameter("sampleRate", "the sampling rate of the audio signal", "(0,inf)", 22050.);
    declareParameter("numberBands", "the number of bands in the filter", "(0,inf)", 6);
    declareParameter("lowFrequencyBound", "the lower bound of the lowest band", "[0,inf)", 20.);
    declareParameter("highFrequencyBound", "the upper bound of the highest band", "(0,inf)", 11000.);
    declareParameter("neighbourRatio", "the ratio of the bins in the sub band used to calculate the peak and valley", "(0,1]", 0.4);
    declareParameter("staticDistribution", "the ratio of the bins to distribute equally", "[0,1]", 0.15);
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

class SpectralContrast : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> >    _spectrum;
  Source<std::vector<Real> >  _spectralcontrast;
  Source<std::vector<Real> >  _valleys;
  std::vector<Real>           _numberOfBinsInBands;
  Real                        _neighbourRatio;
  int                         _startAtBin;

 public:
  SpectralContrast() {
    declareAlgorithm("SpectralContrast");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_spectralcontrast, TOKEN, "spectralContrast");
    declareOutput(_valleys, TOKEN, "spectralValley");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SPECTRALCONTRAST_H
