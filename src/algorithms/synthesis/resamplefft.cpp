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

#include "resamplefft.h"
#include "essentiamath.h"
#include <essentia/utils/synth_utils.h>

using namespace essentia;
using namespace standard;


const char* ResampleFFT::name = "ResampleFFT";
const char* ResampleFFT::category = "Synthesis";
const char* ResampleFFT::description = DOC("This algorithm resamples a sequence using FFT / IFFT. The input and output sizes must be an even number. (It is meant to be eqivalent to the resample function in Numpy).");



// configure
void ResampleFFT::configure()
{
  _fft->configure("size", parameter("inSize").toInt());
  _ifft->configure("size", parameter("outSize").toInt(), "normalize", false);

}


void ResampleFFT::compute() {

  const std::vector<Real>& input = _input.get();
  std::vector<Real>& output = _output.get();


  std::vector<std::complex<Real> >fftin; // temp vectors
  std::vector<std::complex<Real> >fftout; // temp vectors
  std::vector<Real> ifftout; // temp vectors

  int sizeIn = parameter("inSize").toInt(); // input.size();
  int sizeOut = parameter("outSize").toInt();

  _fft->input("frame").set(input);
  _fft->output("fft").set(fftin);
  _fft->compute();

  int hN = (sizeIn/2.)+1;
  int hNout = (sizeOut/2.)+1;
  initializeFFT(fftout, hNout);
  // fill positive spectrum to hN (upsampling zeros will be padded) or hNout (downsampling and high frequencies will be removed)
  for (int i = 0; i < std::min(hN, hNout); ++i)
  {
    // positive spectrums
    fftout[i].real( fftin[i].real());
    fftout[i].imag( fftin[i].imag());
  }

  _ifft->input("fft").set(fftout);
  _ifft->output("frame").set(ifftout);
  _ifft->compute();

  output.resize(0); // clear output

  // normalize
  Real normalizationGain = 1. / float(sizeIn);
  for (int i = 0; i < sizeOut; ++i)
  {
    output.push_back(ifftout[i] * normalizationGain) ;
  }

}



