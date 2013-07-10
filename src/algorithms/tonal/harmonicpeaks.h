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

#ifndef ESSENTIA_HARMONICPEAKS_H
#define ESSENTIA_HARMONICPEAKS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class HarmonicPeaks : public Algorithm {

 protected:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Input<Real> _pitch;
  Output<std::vector<Real> > _harmonicFrequencies;
  Output<std::vector<Real> > _harmonicMagnitudes;

 public:
  HarmonicPeaks() {
    declareInput(_frequencies, "frequencies", "the frequencies of the spectral peaks [Hz] (ascending order)");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the spectral peaks (ascending frequency order)");
    declareInput(_pitch, "pitch", "an estimate of the fundamental frequency of the signal [Hz]");
    declareOutput(_harmonicFrequencies, "harmonicFrequencies", "the frequencies of harmonic peaks [Hz]");
    declareOutput(_harmonicMagnitudes, "harmonicMagnitudes", "the magnitudes of harmonic peaks");
  }

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class HarmonicPeaks : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Sink<Real> _pitch;
  Source<std::vector<Real> > _harmonicFrequencies;
  Source<std::vector<Real> > _harmonicMagnitudes;

 public:
  HarmonicPeaks() {
    declareAlgorithm("HarmonicPeaks");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareInput(_pitch, TOKEN, "pitch");
    declareOutput(_harmonicFrequencies, TOKEN, "harmonicFrequencies");
    declareOutput(_harmonicMagnitudes, TOKEN, "harmonicMagnitudes");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_HARMONICPEAKS_H
