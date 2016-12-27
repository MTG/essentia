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

#include "binaryoperator.h"
#include "essentiamath.h"
#include <sstream>

using namespace essentia;
using namespace standard;

const char* BinaryOperator::name = "BinaryOperator";
const char* BinaryOperator::category = "Standard";
const char* BinaryOperator::description = DOC("This algorithm performs basic arithmetical operations element by element given two arrays.\n"
"Note:\n"
"  - using this algorithm in streaming mode can cause diamond shape graphs which have not been tested with the current scheduler. There is NO GUARANTEE of its correct work for diamond shape graphs.\n"
"  - for y<0, x/y is invalid");

BinaryOperator::OpType BinaryOperator::typeFromString(const std::string& name) const {
  if (name == "add") return ADD;
  if (name == "subtract") return SUBTRACT;
  if (name == "multiply") return MULTIPLY;
  if (name == "divide") return DIVIDE;

  throw EssentiaException("BinaryOperator: Unknown binary operator type: ", name);
}


#define APPLY_FUNCTION(f) {             \
  for (int i=0; i<int(input.size()); ++i) { \
    output[i] = f(input[i]);            \
  }                                     \
  return;                               \
}

void BinaryOperator::compute() {

  const std::vector<Real>& input1 = _input1.get();
  const std::vector<Real>& input2 = _input2.get();
  std::vector<Real>& output = _output.get();
  
  if (input1.size() != input2.size()) {
    throw EssentiaException("BinaryOperator: input vectors are not of equal size");
  }
  output.resize(input1.size());

  switch (_type) {

  case ADD:
    {
      for (size_t i=0; i<input1.size(); ++i) {
        output[i] = input1[i] + input2[i];
      }
      return;
    }

  case SUBTRACT:
    {
      for (size_t i=0; i<input1.size(); ++i) {
        output[i] = input1[i] - input2[i];  
      }
      return;
    }

  case MULTIPLY:
    {
      for (size_t i=0; i<input1.size(); ++i) {
        output[i] = input1[i] * input2[i];
      }
      return;
    }

  case DIVIDE:
    {
      for (size_t i=0; i<input1.size(); ++i) {
        if (!input2[i]) {
          std::ostringstream e ;
          e <<  "BinaryOperator: Divide by zero found in array position " << i;
          throw EssentiaException(e);
        }
        output[i] = input1[i] / input2[i];
      }
      return;
    }
  
  default:
    throw EssentiaException("BinaryOperator: Unknown unary operator type");
  }
}
