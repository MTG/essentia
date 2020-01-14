/*
 * Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_MINMAX_H
#define ESSENTIA_MINMAX_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class MinMax : public Algorithm {

 protected:
  enum OpType {
    MIN,
    MAX
  };

  OpType typeFromString(const std::string& name) const;

  Input<std::vector<Real> > _input;
  Output<Real> _value;
  Output<int> _index;

  OpType _type;

 public:
    MinMax() {
    declareInput(_input, "array", "the input array");
    declareOutput(_value, "real", "the minimum or maximum of the input array, according to the type parameter");
    declareOutput(_index, "int", "the index of the value");
  }

  void declareParameters() {
    declareParameter("type", "the type of the operation", "{min,max}", "min");
  }

  void configure() {
    _type = typeFromString(parameter("type").toString());
  }

  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class MinMax : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _input;
  Source<Real> _value;
  Source<int> _index;

 public:
  MinMax() {
    declareAlgorithm("MinMax");
    declareInput(_input, TOKEN, "array");
    declareOutput(_value, TOKEN, "real");
    declareOutput(_index, TOKEN, "int");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_MINMAX_H
