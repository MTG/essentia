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

#ifndef ESSENTIA_BEATOGRAM_H
#define ESSENTIA_BEATOGRAM_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Beatogram : public Algorithm {

 private:
  Input<std::vector<Real> > _loudness;
  Input<std::vector<std::vector<Real> > > _loudnessBandRatio;
  Output<std::vector<std::vector<Real> > > _beatogram;

  int _windowSize;

 public:
  Beatogram() {
    declareInput(_loudness, "loudness", "the loudness at each beat");
    declareInput(_loudnessBandRatio, "loudnessBandRatio", "matrix of loudness ratios at each band and beat");
    declareOutput(_beatogram, "beatogram", "filtered matrix loudness");
  }

  ~Beatogram() {}

  void declareParameters() {
    declareParameter("size", "number of beats for dynamic filtering", "[1,inf)", 16);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* version;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Beatogram : public StreamingAlgorithmWrapper {

  // TODO: it should be implemented apropriately for a pure streaming algo

 protected:
  Sink<std::vector<Real> > _loudness;
  Sink<std::vector<std::vector<Real> > > _loudnessBandRatio;
  Source<std::vector<std::vector<Real> > > _beatogram;

 public:
  Beatogram() {
    declareAlgorithm("Beatogram");
    declareInput(_loudness, TOKEN, "loudness");
    declareInput(_loudnessBandRatio, TOKEN, "loudnessBandRatio");
    declareOutput(_beatogram, TOKEN, "beatogram");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BEATOGRAM_H
