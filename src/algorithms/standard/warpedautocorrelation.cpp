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

#include "warpedautocorrelation.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* WarpedAutoCorrelation::name = "WarpedAutoCorrelation";
const char* WarpedAutoCorrelation::description = DOC("This algorithm returns the warped auto-correlation of an audio signal. The implementation is an adapted version of K. Schmidt's implementation of the matlab algorithm from the 'warped toolbox' by Aki Harma and Matti Karjalainen found [2]. For a detailed explanation of the algorithm, see [1].\n"
"This algorithm is only defined for positive lambda = 1.0674*sqrt(2.0*atan(0.00006583*sampleRate)/PI) - 0.1916, thus it will throw an exception when the supplied sampling rate does not pass the requirements.\n"
"If maxLag is larger than the size of the input array, an exception is thrown.\n"
"\n"
"References:\n"
"  [1] A. Härmä, M. Karjalainen, L. Savioja, V. Välimäki, U. K. Laine, and\n"
"  J. Huopaniemi, \"Frequency-Warped Signal Processing for Audio Applications,\"\n"
"  JAES, vol. 48, no. 11, pp. 1011–1031, 2000.\n\n"
"  [2] WarpTB - Matlab Toolbox for Warped DSP\n"
"  http://www.acoustics.hut.fi/software/warp");

void WarpedAutoCorrelation::configure() {

  Real sampleRate = parameter("sampleRate").toReal();
  _lambda = 1.0674*sqrt(2.0*atan(0.00006583*sampleRate)/M_PI) - 0.1916;

  if (fabs(_lambda) >= 1.0)
    throw EssentiaException("WarpedAutoCorrelation: invalid sampling rate given");
}

void WarpedAutoCorrelation::compute() {

  const std::vector<Real>& signal = _signal.get();
  std::vector<Real>& warpedAutoCorrelation = _warpedAutoCorrelation.get();

  int maxLag = parameter("maxLag").toInt();

  if (maxLag >= int(signal.size())) {
    throw EssentiaException("WarpedAutoCorrelation: maxLag is not smaller than the input signal size");
  }

  warpedAutoCorrelation.resize(maxLag);

  // copy the input to tmp
  _tmp = signal;

  // don't remove the 'std::', it will use the wrong fill otherwise
  std::fill(warpedAutoCorrelation.begin(), warpedAutoCorrelation.end(), Real(0.0));

  // The following implementation matches exactly the following system, which
  // is exactly what is in the paper:
  //
  //
  //                          ,---<--- {Z^-1} ---<--.
  //                          |                     |
  //  _tmp[i-1] (filtered) -> |                     |
  //                          |                     |
  //                          |                     |
  //  _tmp[i] -->-- {*-1} --- + -- {*_lambda} -- + ------>-- _tmp[i] (now filtered)
  //             |                               ^
  //             v                               |
  //             |                               |   <- previous_in
  //             `-------- {Z^-1}------>---------'
  //
  //

  for(int lag=0; lag<maxLag; ++lag) {
    Real previous_in = 0.0;

    for(int i=0; i<int(signal.size()); ++i) {
      // the auto correlation
      warpedAutoCorrelation[lag] += _tmp[i] * signal[i];

      // warp the correlation vector by applying the allpass filter
      Real tmp;
      if (i==0) tmp = -_lambda * _tmp[i];
      else tmp = (_tmp[i-1] - _tmp[i])*_lambda + previous_in;
      previous_in = _tmp[i];
      _tmp[i] = tmp;
    }
  }
}
