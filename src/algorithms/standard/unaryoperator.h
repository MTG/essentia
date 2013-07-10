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

#ifndef ESSENTIA_UNARYOPERATOR_H
#define ESSENTIA_UNARYOPERATOR_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class UnaryOperator : public Algorithm {

 protected:
  enum OpType {
    IDENTITY,
    ABS,
    LOG10,
    LN,
    LIN2DB,
    DB2LIN,
    SIN,
    COS,
    SQRT,
    SQUARE
  };

  OpType typeFromString(const std::string& name) const;

  Input<std::vector<Real> > _input;
  Output<std::vector<Real> > _output;

  OpType _type;

 public:
  UnaryOperator() {
    declareInput(_input, "array", "the input array");
    declareOutput(_output, "array", "the input array transformed by unary operation");
  }

  void declareParameters() {
    declareParameter("type", "the type of the unary operator to apply to input array", "{identity,abs,log10,log,ln,lin2db,db2lin,sin,cos,sqrt,square}", "identity");
  }

  void configure() {
    _type = typeFromString(parameter("type").toString());
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

class UnaryOperator : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _input;
  Source<std::vector<Real> > _output;

 public:
  UnaryOperator() {
    declareAlgorithm("UnaryOperator");
    declareInput(_input, TOKEN, "array");
    declareOutput(_output, TOKEN, "array");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_UNARYOPERATOR_H
