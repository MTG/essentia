/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
