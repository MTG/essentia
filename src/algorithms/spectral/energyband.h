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

#ifndef ESSENTIA_ENERGYBAND_H
#define ESSENTIA_ENERGYBAND_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class EnergyBand : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _energyBand;

  Real _normStartIdx, _normStopIdx;

 public:
  EnergyBand() {
    declareInput(_spectrum, "spectrum", "the input frequency spectrum");
    declareOutput(_energyBand, "energyBand", "the energy in the frequency band");
  }

  void declareParameters() {
    declareParameter("startCutoffFrequency", "the start frequency from which to sum the energy [Hz]", "[0,inf)", 0.0);
    declareParameter("stopCutoffFrequency", "the stop frequency to which to sum the energy [Hz]", "(0,inf)", 100.0);
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace essentia
} // namespace standard

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class EnergyBand : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _energyBand;

 public:
  EnergyBand() {
    declareAlgorithm("EnergyBand");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_energyBand, TOKEN, "energyBand");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ENERGYBAND_H
