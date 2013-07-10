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

#ifndef ESSENTIA_SILENCERATE_H
#define ESSENTIA_SILENCERATE_H


#include "algorithm.h"

namespace essentia {
namespace standard {

class SilenceRate : public Algorithm {

 protected:
  Input<std::vector<Real> > _frame;
  std::vector<Output<Real>*> _outputs;

  std::vector<Real> _thresholds;

 public:
  SilenceRate() {
    declareInput(_frame, "frame", "the input frame");
  }

  ~SilenceRate() {}

  void declareParameters() {
    declareParameter("thresholds", "the threshold values", "", std::vector<Real>());
  }

  void configure();

  void compute();
  void clearOutputs();
  void reset() {}

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

/**
 * @todo make sure this algo doesn't break when configuring more than once.
 *       (it most certainly does, right now)
 */
class SilenceRate : public Algorithm {
 protected:
  Sink<std::vector<Real> > _frame;
  std::vector<Source<Real>*> _outputs;

  std::vector<Real> _thresholds;

  void clearOutputs();

 public:

  SilenceRate() : Algorithm() {
    declareInput(_frame, 1, "frame", "the input frame");
  }

  ~SilenceRate() {
    clearOutputs();
  }

  void declareParameters() {
    declareParameter("thresholds", "the threshold values", "", std::vector<Real>());
  }

  void configure();

  AlgorithmStatus process();

  static const char* name;
  static const char* description;

};


} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SILENCERATE_H
