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

#include "binaryoperator.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* BinaryOperator::name = "BinaryOperator";
const char* BinaryOperator::description = DOC("Given two vectors of equal size, this algorithm performs basic arithmetic operations element by element. Note that if using the division operator, '/', input2 cannot contain zeros.\n"
"Note that input vectors must be of the same size and that division by zero is not defined. An exception will be thrown if any of these cases occurs.");

void BinaryOperator::compute() {

  const vector<Real>& input1 = _input1.get();
  const vector<Real>& input2 = _input2.get();
  vector<Real>& output = _output.get();

  if (input1.size() != input2.size()) {
    throw EssentiaException("BinaryOperator: input vectors have different size");
  }

  output.resize(input1.size());

  string operatorType = parameter("type").toString();

  if (operatorType == "+") {
    for (int i=0; i<int(input1.size()); ++i) {
      output[i] = input1[i] + input2[i];
    }
    return;
  }

  if (operatorType == "-") {
    for (int i=0; i<int(input1.size()); ++i) {
      output[i] = input1[i] - input2[i];
    }
    return;
  }

  if (operatorType == "*") {
    for (int i=0; i<int(input1.size()); ++i) {
      output[i] = input1[i] * input2[i];
    }
    return;
  }

  if (operatorType == "/") {
    for (int i=0; i<int(input1.size()); ++i) {
      if (input2[i] == 0) {
        throw EssentiaException("BinaryOperator: input2 contains zeros, cannot perform '/' operation when right-hand side input contains zeros");
      }
      output[i] = input1[i] / input2[i];
    }
    return;
  }

  throw EssentiaException("BinaryOperator: Unimplemented operator type: '", operatorType, "'");
}
