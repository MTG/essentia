/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_SDA_H
#define ESSENTIA_SDA_H

#include "algorithm.h"
#include "pool.h"
#include <tensorflow/c/c_api.h>
#include <3rdparty/boost_1_68_0/boost/multi_array.hpp>


namespace essentia {
namespace standard {

class SDA : public Algorithm {

 protected:
  Input<boost::multi_array<Real, 3> > _input;
  Output<boost::multi_array<Real, 3> > _output;



 public:
  SDA() {
    declareInput(_input, "poolIn", "the pool where to get the feature tensors");
    declareOutput(_output, "poolOut", "the pool where to store the predicted tensors");
  }

  void declareParameters() {};

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} //namespace standard
} //namespace essentia

#endif // ESSENTIA_SDA_H
