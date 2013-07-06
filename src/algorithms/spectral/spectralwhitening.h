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

#ifndef SPECTRALWHITENING_H
#define SPECTRALWHITENING_H

#include "algorithm.h"
#include "bpfutil.h"

namespace essentia {
namespace standard {

class SpectralWhitening : public Algorithm {

 protected:
  Input< std::vector<Real> > _spectrum;
  Input< std::vector<Real> > _frequencies;
  Input< std::vector<Real> > _magnitudes;
  Output< std::vector<Real> > _magnitudesWhite;

  Real _maxFreq;
  Real _spectralRange;

  essentia::util::BPF _noiseBPF;

 public:
  SpectralWhitening() {
    declareInput(_spectrum, "spectrum", "the audio linear spectrum");
    declareInput(_frequencies, "frequencies", "the spectral peaks' linear frequencies");
    declareInput(_magnitudes, "magnitudes", "the spectral peaks' linear magnitudes");
    declareOutput(_magnitudesWhite, "magnitudes", "the whitened spectral peaks' linear magnitudes");
  }

  ~SpectralWhitening() {
  }

  void declareParameters() {
    declareParameter("maxFrequency", "max frequency to apply whitening to [Hz]", "(0,inf)", 5000.0);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

  static const Real bpfResolution;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class SpectralWhitening : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<std::vector<Real> > _magnitudesWhite;

 public:
  SpectralWhitening() {
    declareAlgorithm("SpectralWhitening");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_magnitudesWhite, TOKEN, "magnitudes");
  }
};

} // namespace streaming
} // namespace essentia

#endif // SPECTRALWHITENING_H
