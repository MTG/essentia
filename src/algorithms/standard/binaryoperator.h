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

#ifndef ESSENTIA_BINARYOPERATOR_H
#define ESSENTIA_BINARYOPERATOR_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class BinaryOperator : public Algorithm {

 private:
  Input<std::vector<Real> > _input1;
  Input<std::vector<Real> > _input2;
  Output<std::vector<Real> > _output;

 public:
  BinaryOperator() {
    declareInput(_input1, "input1", "the left-hand side input");
    declareInput(_input2, "input2", "the right-hand side input");
    declareOutput(_output, "output", "the resulting vector");
  }

  void declareParameters() {
    declareParameter("type", "the type of the operation", "{+,-,*,/}", "+");
  }

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class BinaryOperator : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _input1;
  Sink<std::vector<Real> > _input2;
  Source<std::vector<Real> > _output;

 public:
  BinaryOperator() {
    declareAlgorithm("BinaryOperator");
    declareInput(_input1, TOKEN, "input1");
    declareInput(_input2, TOKEN, "input2");
    declareOutput(_output, TOKEN, "output");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BINARYOPERATOR_H
