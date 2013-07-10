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

#ifndef ESSENTIA_ENERGYBANDRATIO_H
#define ESSENTIA_ENERGYBANDRATIO_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class EnergyBandRatio : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _energyBandRatio;

  Real _startFreqNormalized, _stopFreqNormalized;

 public:
  EnergyBandRatio() {
    declareInput(_spectrum, "spectrum", "the input audio spectrum");
    declareOutput(_energyBandRatio, "energyBandRatio", "the energy ratio of the specified band over the total energy");
  }

  void declareParameters() {
    declareParameter("startFrequency", "the frequency from which to start summing the energy [Hz]", "[0,inf)", 0.0);
    declareParameter("stopFrequency", "the frequency up to which to sum the energy [Hz]", "[0,inf)", 100.0);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
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

class EnergyBandRatio : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _energyBandRatio;

 public:
  EnergyBandRatio() {
    declareAlgorithm("EnergyBandRatio");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_energyBandRatio, TOKEN, "energyBandRatio");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_ENERGYBANDRATIO_H
