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

#ifndef ESSENTIA_INSTANTPOWER_H
#define ESSENTIA_INSTANTPOWER_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class InstantPower : public Algorithm {

 private:
  Input<std::vector<Real> > _array;
  Output<Real> _power;

 public:
  InstantPower() {
    declareInput(_array, "array", "the input array");
    declareOutput(_power, "power", "the instant power of the input array");
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

class InstantPower : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<Real> _rms;

 public:
  InstantPower() {
    declareAlgorithm("InstantPower");
    declareInput(_array, TOKEN, "array");
    declareOutput(_rms, TOKEN, "power");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_INSTANTPOWER_H
