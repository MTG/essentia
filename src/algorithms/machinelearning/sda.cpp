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

#include "sda.h"
#include "algorithmfactory.h"

using namespace std;
using namespace essentia;
using namespace standard;
using namespace boost;

const char* SDA::name = "SDA";
const char* SDA::category = "Machine Learning";
const char* SDA::description = DOC("dfsdfdsfsdaf");


void SDA::configure() {
};


void SDA::compute() {
  // const Real& input = _input.get();
  const_multi_array_ref<Real, 3> input(_input.get());
  multi_array<Real, 3>& output = _output.get();
  // Real& output = _output.get();
  auto& input_shape = reinterpret_cast<boost::array<size_t, const_multi_array_ref<Real, 3>::dimensionality> const&>(*input.shape());



  // output.resize(input_shape);
  // output.reshape(input_shape);
  // output = input;

  multi_array<Real, 3> intermediate(boost::extents[3][3][3]);
  // auto& input_shape = reinterpret_cast<boost::array<size_t, const_multi_array_ref<Real, 3>::dimensionality> const&>(*intermediate.shape());

  output.resize(input_shape);
  output.reshape(input_shape);
  output = input;

  output[0][0][2] = 5.f;
}

