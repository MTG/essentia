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

#include "stochasticmodelsynth.h"
#include "essentiamath.h"
#include <essentia/utils/synth_utils.h>

using namespace essentia;
using namespace standard;

const char* StochasticModelSynth::name = "StochasticModelSynth";
const char* StochasticModelSynth::category = "Synthesis";
const char* StochasticModelSynth::description = DOC("This algorithm computes the stochastic model synthesis. It generates the noisy spectrum from a resampled spectral envelope of the stochastic component.\n"
"\n"
"References:\n"
"  https://github.com/MTG/sms-tools\n"
"  http://mtg.upf.edu/technologies/sms\n"
);



void StochasticModelSynth::configure() {

  _stocf = parameter("stocf").toReal();
  _fftSize = parameter("fftSize").toInt();
  _hopSize = parameter("hopSize").toInt();

  _window->configure("type", "hann", "size", _fftSize);
  _ifft->configure("size", _fftSize );

  // resample for stochastic envelope using FFT
  _hN = int(_fftSize/2.) + 1;
  _stocf = std::max(_stocf, 3.f / _hN); //  limits  Stochastic decimation factor too small

  _stocSize = int (_fftSize * _stocf / 2.);
  // adapt resampleFFT data for even input and output sizes
  _stocSize += _stocSize % 2;
  _resample->configure("inSize", _stocSize, "outSize", _hN + 1);

  _overlapadd->configure("frameSize", _fftSize, // uses synthesis window
                         "hopSize", parameter("hopSize").toInt());
}



void StochasticModelSynth::compute() {

  // inputs and outputs
 const  std::vector<Real>& stocEnv = _stocenv.get();
 std::vector<Real>& frame = _frame.get();


  std::vector<Real> magResDB;
  std::vector<std::complex<Real> > fftMagRes;
  std::vector<Real> ifftframe;
  std::vector<Real> wframe;


  // limit size of input envelope before resampling
  std::vector<Real> stocEnv2 = stocEnv;

  if (_stocSize < (int) stocEnv2.size())
  {
   stocEnv2.erase (stocEnv2.begin()+_stocSize, stocEnv2.end());
  }

  _resample->input("input").set(stocEnv2);
  _resample->output("output").set(magResDB);
  _resample->compute();

  // adapt size of input spectral envelope and resampled vector (FFT algorihm requires even sizes)
  if ((int) magResDB.size() > _hN)
    magResDB.pop_back(); // remove last value

  getFFTFromEnvelope(magResDB, fftMagRes);

  _ifft->input("fft").set(fftMagRes);
  _ifft->output("frame").set(ifftframe);
  _ifft->compute();

  // synthesis window
  // frame is of size 2*hopsize
  _window->input("frame").set(ifftframe);
  _window->output("frame").set(wframe);
  _window->compute();

	// overlapp add synthesized audio
	_overlapadd->input("signal").set(wframe);
	_overlapadd->output("signal").set(frame);
	_overlapadd->compute();

}


// ---------------------------
// additional methods

void StochasticModelSynth::getFFTFromEnvelope(const std::vector<Real> magResDB, std::vector<std::complex<Real> > &fftStoc)
{
  // get spectral envelope in DB
  Real magdB, phase;
  int N = (int)magResDB.size();

  initializeFFT(fftStoc,N);
  Real scale = Real(_fftSize)/2.f; // normalization to match stochastic analysis input energy.

  for (int i = 0; i < N; ++i)
  {
    phase =  2 * M_PI *  Real(rand()/Real(RAND_MAX));
    magdB = magResDB[i];

    // positive spectrums
    fftStoc[i].real(scale * powf(10.f, (magdB / 20.f)) * cos(phase) ) ;
    fftStoc[i].imag(scale * powf(10.f, (magdB / 20.f)) * sin(phase) ) ;
  }

}


