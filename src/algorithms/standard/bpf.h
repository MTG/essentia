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

#ifndef ESSENTIA_BPF_H
#define ESSENTIA_BPF_H

#include "algorithm.h"
#include "bpfutil.h"

namespace essentia {
namespace standard {

class BPF : public Algorithm {

 protected:
  Input<Real> _xInput;
  Output<Real> _yOutput;

  essentia::util::BPF bpf;
 public:
  BPF() {
    declareInput(_xInput, "x", "the input coordinate (x-axis)");
    declareOutput(_yOutput, "y", "the output coordinate (y-axis)");
  }

  ~BPF() {}

  void declareParameters() {
    std::vector<Real> defaultPoints(2);
    defaultPoints[0] = 0;
    defaultPoints[1] = 1;
    declareParameter("xPoints", "the x-coordinates of the points forming the break-point function (the points must be arranged in ascending order and cannot contain duplicates)", "", defaultPoints);
    declareParameter("yPoints", "the y-coordinates of the points forming the break-point function", "", defaultPoints);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;
};

} // namespace essentia
} // namespace standard

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class BPF : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _xInput;
  Source<Real> _yOutput;

 public:
  BPF() {
    declareAlgorithm("BPF");
    declareInput(_xInput, TOKEN, "x");
    declareOutput(_yOutput, TOKEN, "y");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BPF_H
