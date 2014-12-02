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

#ifndef ESSENTIA_SPECTRALPEAKS_H
#define ESSENTIA_SPECTRALPEAKS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class SpectralPeaks : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrum;
  Output<std::vector<Real> > _magnitudes;
  Output<std::vector<Real> > _frequencies;
  Algorithm* _peakDetect;

 public:
  SpectralPeaks() {
    declareInput(_spectrum, "spectrum", "the input spectrum");
    declareOutput(_frequencies, "frequencies", "the frequencies of the spectral peaks [Hz]");
    declareOutput(_magnitudes, "magnitudes", "the magnitudes of the spectral peaks");

    _peakDetect = AlgorithmFactory::create("PeakDetection");
  }

  ~SpectralPeaks() {
    delete _peakDetect;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("maxPeaks", "the maximum number of returned peaks", "[1,inf)", 100);
    declareParameter("maxFrequency", "the maximum frequency of the range to evaluate [Hz]", "(0,inf)", 5000.0);
    declareParameter("minFrequency", "the minimum frequency of the range to evaluate [Hz]", "[0,inf)", 0.0);
    declareParameter("magnitudeThreshold", "peaks below this given threshold are not outputted", "(-inf,inf)", 0.0);
    declareParameter("orderBy", "the ordering type of the outputted peaks (ascending by frequency or descending by magnitude)", "{frequency,magnitude}", "frequency");
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

class SpectralPeaks : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<std::vector<Real> > _frequencies;
  Source<std::vector<Real> > _magnitudes;

 public:
  SpectralPeaks() {
    declareAlgorithm("SpectralPeaks");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_frequencies, TOKEN, "frequencies");
    declareOutput(_magnitudes, TOKEN, "magnitudes");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SPECTRALPEAKS_H
