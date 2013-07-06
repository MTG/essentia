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

#include "unaryoperator.h"
#include "essentiamath.h"
#include <sstream>

using namespace essentia;
using namespace standard;

const char* UnaryOperator::name = "UnaryOperator";
const char* UnaryOperator::description = DOC("Given a vector of Reals, this algorithm will perform basic arithmetical operations on it, element by element.\n"
"Note:\n"
"  - log and ln are equivalent to the natural logarithm\n"
"  - for log, ln, log10 and lin2db, x is clipped to 1e-30 for x<1e-30\n"
"  - for x<0, sqrt(x) is invalid");

UnaryOperator::OpType UnaryOperator::typeFromString(const std::string& name) const {
  if (name == "identity") return IDENTITY;
  if (name == "abs") return ABS;
  if (name == "log10") return LOG10;
  if (name == "log") return LN;
  if (name == "ln") return LN;
  if (name == "lin2db") return LIN2DB;
  if (name == "db2lin") return DB2LIN;
  if (name == "sin") return SIN;
  if (name == "cos") return COS;
  if (name == "sqrt") return SQRT;
  if (name == "square") return SQUARE;

  throw EssentiaException("UnaryOperator: Unknown unary operator type: ", name);
}

inline Real square_func(Real x) {
  return x*x;
}

#define APPLY_FUNCTION(f) {             \
  for (int i=0; i<int(input.size()); ++i) { \
    output[i] = f(input[i]);            \
  }                                     \
  return;                               \
}

void UnaryOperator::compute() {

  const std::vector<Real>& input = _input.get();
  std::vector<Real>& output = _output.get();

  output.resize(input.size());

  switch (_type) {

  case IDENTITY:
    output = input;
    return;

  case ABS: APPLY_FUNCTION(fabs);

  case LOG10:
    {
      Real cutoff = 1e-30;
      for (int i=0; i<int(input.size()); ++i) {
        if (input[i] < cutoff) {
          output[i] = log10(cutoff);
        }
        else {
          output[i] = log10(input[i]);
        }
      }
      return;
    }

  case LN:
    {
      Real cutoff = 1e-30;
      for (int i=0; i<int(input.size()); ++i) {
        if (input[i] < cutoff) {
          output[i] = log(cutoff);
        }
        else {
          output[i] = log(input[i]);
        }
      }
      return;
    }

  case LIN2DB: APPLY_FUNCTION(lin2db);
  case DB2LIN: APPLY_FUNCTION(db2lin);
  case SIN:    APPLY_FUNCTION(sin);
  case COS:    APPLY_FUNCTION(cos);
  case SQRT:
    {
      for (int i=0; i<(int)input.size(); i++) {
        if (input[i] < 0) {
          std::ostringstream e ;
          e <<  "UnaryOperator: Cannot compute sqrt(" << input[i] << "). Found in array position " << i;
          throw EssentiaException(e);
        }
        output[i] = sqrt(input[i]);
      }
      return;
    }

  case SQUARE: APPLY_FUNCTION(square_func);

  default:
    throw EssentiaException("UnaryOperator: Unknown unary operator type");
  }
}
