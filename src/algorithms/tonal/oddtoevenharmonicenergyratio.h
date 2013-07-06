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

#ifndef ESSENTIA_ODDTOEVENHARMONICENERGYRATIO_H
#define ESSENTIA_ODDTOEVENHARMONICENERGYRATIO_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class OddToEvenHarmonicEnergyRatio : public Algorithm {

 protected:
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _magnitudes;
  Output<Real> _oddtoevenharmonicenergyratio;

 public:
  OddToEvenHarmonicEnergyRatio() {
    declareInput(_frequencies, "frequencies", "the frequencies of the harmonic peaks (at least two frequencies in frequency ascending order)");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the harmonic peaks (at least two magnitudes in frequency ascending order)");
    declareOutput(_oddtoevenharmonicenergyratio, "oddToEvenHarmonicEnergyRatio", "the ratio between the odd and even harmonic energies of the given harmonic peaks");
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

class OddToEvenHarmonicEnergyRatio : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _magnitudes;
  Source<Real> _oddtoevenharmonicenergyratio;

 public:
  OddToEvenHarmonicEnergyRatio() {
    declareAlgorithm("OddToEvenHarmonicEnergyRatio");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_oddtoevenharmonicenergyratio, TOKEN, "oddToEvenHarmonicEnergyRatio");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ODDTOEVENHARMONICENERGYRATIO_H
