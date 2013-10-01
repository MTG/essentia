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

#include "iir.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* IIR::name = "IIR";
const char* IIR::description = DOC("This algorithm implements a standard IIR filter. It filters the data in the input vector with the filter described by parameter vectors 'numerator' and 'denominator' to create the output filtered vector. In the litterature, the numerator is often referred to as the 'B' coefficients and the denominator as the 'A' coefficients.\n"
"\n"
"The filter is a Direct Form II Transposed implementation of the standard difference equation:\n"
"  a(0)*y(n) = b(0)*x(n) + b(1)*x(n-1) + ... + b(nb-1)*x(n-nb+1) - a(1)*y(n-1) - ... - a(nb-1)*y(n-na+1)\n"
"\n"
"This algorithm maintains a state which is the state of the delays. One should call the reset() method to reinitialize the state to all zeros.\n"
"\n"
"An exception is thrown if the \"numerator\" or \"denominator\" parameters are empty. An exception is also thrown if the first coefficient of the \"denominator\" parameter is 0.\n"
"\n"
"References:\n"
"  [1] Smith, J.O.  Introduction to Digital Filters with Audio Applications,\n" 
"  http://ccrma-www.stanford.edu/~jos/filters/\n\n"
"  [2] Infinite Impulse Response - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/IIR");

void IIR::reset() {
  for (int i=0; i<int(_state.size()); ++i) {
    _state[i] = 0.0;
  }
}

void IIR::configure() {
  _a = parameter("denominator").toVectorReal();
  _b = parameter("numerator").toVectorReal();

  if (_b.empty()) {
    throw EssentiaException("IIR: the numerator vector is empty");
  }
  if (_a.empty()) {
    throw EssentiaException("IIR: the denominator vector is empty");
  }
  if (_a[0] == 0.0) {
    throw EssentiaException("IIR: the first coefficient of the denominator vector must not be 0");
  }

  // normalize everything with a[0]
  for (int i=1; i<int(_a.size()); ++i) {
    _a[i] /= _a[0];
  }

  for (int i=0; i<int(_b.size()); ++i) {
    _b[i] /= _a[0];
  }

  _a[0] = 1.0;

  int wantedSize = int(max(_b.size(), _a.size()));
  if (int(_state.size()) != wantedSize) {
    _state.resize(wantedSize);
    reset();
  }
}

// prevent denormalization (in IIR filter feedback loop, for instance)
// adding a constant epsilon to the values in the state line is a tad
// faster (~6%), but I (nwack) like this method better as it is more
// correct, and if fed with 0, will return 0 as well (not epsilon)
inline void renormalize(Real& x) {
  if (isDenormal(x)) {
    x = Real(0.0);
  }
}


inline void updateStateLine(vector<Real>& state, int size,
                            const vector<Real>& a, const vector<Real>& b,
                            const Real& x, const Real& y) {
  for (int k=1; k<size; ++k) {
    state[k-1] = (b[k]*x - a[k]*y) + state[k];
    renormalize(state[k-1]);
  }
}

template <int n>
void updateStateLineUnrolled(vector<Real>& state,
                            const vector<Real>& a, const vector<Real>& b,
                            const Real& x, Real& y) {
  for (int k=1; k<n; ++k) {
    state[k-1] = b[k]*x - a[k]*y + state[k];
  }

  // make sure there are no denormal numbers in the state line
  for (int k=1; k<n; ++k) {
    renormalize(state[k-1]);
  }

}

template <int filterSize>
void filterABEqualSize(const vector<Real>& x, vector<Real>& y,
                       const vector<Real>& a, const vector<Real>& b,
                       vector<Real>& state) {
  for (int n=0; n < int(y.size()); ++n) {
    y[n] = b[0]*x[n] + state[0];
    updateStateLineUnrolled<filterSize>(state, a, b, x[n], y[n]);
  }
}



void IIR::compute() {

  const vector<Real>& x = _x.get();
  vector<Real>& y = _y.get();

  y.resize(x.size());

  if (_b.size() == _a.size()) {
    switch (_a.size()) {

    case 2:  filterABEqualSize<2> (x, y, _a, _b, _state); break;
    case 3:  filterABEqualSize<3> (x, y, _a, _b, _state); break;
    case 4:  filterABEqualSize<4> (x, y, _a, _b, _state); break;
    case 5:  filterABEqualSize<5> (x, y, _a, _b, _state); break;
    case 6:  filterABEqualSize<6> (x, y, _a, _b, _state); break;
    case 7:  filterABEqualSize<7> (x, y, _a, _b, _state); break;
    case 8:  filterABEqualSize<8> (x, y, _a, _b, _state); break;
    case 9:  filterABEqualSize<9> (x, y, _a, _b, _state); break;
    case 10: filterABEqualSize<10>(x, y, _a, _b, _state); break;
    case 11: filterABEqualSize<11>(x, y, _a, _b, _state); break;
    case 12: filterABEqualSize<12>(x, y, _a, _b, _state); break;
    case 13: filterABEqualSize<13>(x, y, _a, _b, _state); break;
    case 14: filterABEqualSize<14>(x, y, _a, _b, _state); break;
    case 15: filterABEqualSize<15>(x, y, _a, _b, _state); break;
    case 16: filterABEqualSize<16>(x, y, _a, _b, _state); break;

    default:
      for (int n=0; n < int(y.size()); ++n) {
        y[n] = _b[0]*x[n] + _state[0];
        updateStateLine(_state, _state.size(), _a, _b, x[n], y[n]);
      }
    }
  }

  else if (_b.size() > _a.size()) {
    for (int n=0; n < int(y.size()); ++n) {
      y[n] = _b[0]*x[n]  + _state[0];
      updateStateLine(_state, _a.size(), _a, _b, x[n], y[n]);

      for (int k=_a.size(); k < int(_state.size()); ++k) {
        _state[k-1] = _b[k]*x[n]  + _state[k];
        renormalize(_state[k-1]);
      }
    }
  }

  else { //if (a.size() > b.size()) {
    for (int n=0; n < int(y.size()); ++n) {
      y[n] = _b[0]*x[n]  + _state[0];
      updateStateLine(_state, _b.size(), _a, _b, x[n], y[n]);

      for (int k=_b.size(); k < int(_state.size()); ++k) {
        _state[k-1] = (-_a[k]*y[n])  + _state[k];
        renormalize(_state[k-1]);
      }
    }
  }
}
