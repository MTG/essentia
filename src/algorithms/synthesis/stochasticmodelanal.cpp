/*
 * Copyright (C) 2006-2015  Music Technology Group - Universitat Pompeu Fabra
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

#include "stochasticmodelanal.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* StochasticModelAnal::name = "StochasticModelAnal";
const char* StochasticModelAnal::category = "Synthesis";
const char* StochasticModelAnal::description = DOC("This algorithm computes the stochastic model analysis. It gets the resampled spectral envelope of the stochastic component.\n"
"\n"
"References:\n"
"  https://github.com/MTG/sms-tools\n"
"  http://mtg.upf.edu/technologies/sms\n"
);



void StochasticModelAnal::configure() {

  _stocf = parameter("stocf").toReal();
  _fftSize = parameter("fftSize").toInt();

  _window->configure("type", "hann", "size", _fftSize);
  _fft->configure("size", _fftSize );

  // resample for stochastic envelope using FFT
  _hN = int(_fftSize/2.) + 1;
  _stocf = std::max(_stocf, 3.f / _hN); //  limits  Stochastic decimation factor too small

  _stocSize = int (_fftSize * _stocf / 2.);
  _stocSize += _stocSize % 2;

  _resample->configure("inSize", _hN + 1, "outSize", _stocSize);

}



void StochasticModelAnal::compute() {
  // inputs and outputs
  const std::vector<Real>& frame = _frame.get();
  std::vector<Real>& stocEnv = _stocenv.get();

  std::vector<Real> wframe;
  std::vector<std::complex<Real> > fftin;
  std::vector<Real> magResDB;


  // frame is of size 2*hopsize
  _window->input("frame").set(frame);
  _window->output("frame").set(wframe);
  _window->compute();

  _fft->input("frame").set(wframe);
  _fft->output("fft").set(fftin);
  _fft->compute();

  getSpecEnvelope(fftin, magResDB);

  if ((int) magResDB.size() <  _hN+1)
    magResDB.push_back(magResDB[magResDB.size()-1]); // copy last value

   // adapt size of input spectral envelope and resampled vector (FFT algorihm requires even sizes)
  _resample->input("input").set(magResDB);
  _resample->output("output").set(stocEnv);
  _resample->compute();


}


// ---------------------------
// additional methods

void StochasticModelAnal::getSpecEnvelope(const std::vector<std::complex<Real> > fftRes,std::vector<Real> &magResDB)
{

// get spectral envelope in DB
 Real mag, magdB;

 for (int i=0; i< (int) fftRes.size(); i++)
  {
    // compute fft abs
    mag =  sqrt( fftRes[i].real() * fftRes[i].real() +  fftRes[i].imag() * fftRes[i].imag());
    magdB = std::max(-200., 20. * log10( mag + 1e-10));
    magResDB.push_back(magdB);
  }
}



