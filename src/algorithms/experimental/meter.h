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

#ifndef ESSENTIA_METER_H
#define ESSENTIA_METER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Meter : public Algorithm {

 private:
  Input<std::vector<std::vector<Real> > > _beatogram;
  Output<Real> _meter;

 public:
  Meter() {
    declareInput(_beatogram, "beatogram", "filtered matrix loudness");
    declareOutput(_meter, "meter", "the time signature");
  }

  ~Meter() {}

  void declareParameters() {}

  void compute();
  void configure();

  static const char* name;
  static const char* version;
  static const char* description;

 private:
  bool isPowerN(int val, int power);
  bool isPowerHarmonic(int x, int y);
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Meter : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::vector<Real> > > _beatogram;
  Source<Real> _meter;

 public:
  Meter() {
    declareAlgorithm("Meter");
    declareInput(_beatogram, TOKEN, "beatogram");
    declareOutput(_meter, TOKEN, "meter");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_METER_H
