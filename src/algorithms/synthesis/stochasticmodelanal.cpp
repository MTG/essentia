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

#include "stochasticmodelanal.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* StochasticModelAnal::name = "StochasticModelAnal";
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
  //const std::vector<std::complex<Real> >& fft = _fft.get();
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

  if (magResDB.size() < (int) _hN+1)
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

/*
void StochasticModelAnal::stochasticModelAnalOld(const std::vector<std::complex<Real> > fftInput, const std::vector<Real> magnitudes, const std::vector<Real> frequencies, const std::vector<Real> phases, std::vector<Real> &stocEnv)
{

// TOD: refactor this function in two new essentia algorithms: sineSubctraction and sotchasticModelAnal

  // subtract sines
  std::vector<std::complex<Real> > fftSines;
  std::vector<std::complex<Real> > fftRes;

  _sineModelSynth->input("magnitudes").set(magnitudes);
  _sineModelSynth->input("frequencies").set(frequencies);
  _sineModelSynth->input("phases").set(phases);
  _sineModelSynth->output("fft").set(fftSines);

  _sineModelSynth->compute();

  fftRes = fftInput; // initialize output


  for (int i= 0; i < (int)fftRes.size(); ++i)
  {
    fftRes[i].real(fftInput[i].real() - fftSines[i].real());
    fftRes[i].imag(fftInput[i].imag() - fftSines[i].imag());
  }

  // the decimation factor must be in a range (0.01 and 1) Default 0 0.2
  Real stocf = std::min( std::max(0.01f, parameter("stocf").toReal()), 1.f);
  // To obtain the stochastic envelope, we resample only half of the FFT size (i.e. fftRes.size()-1)
  int stocSize =  int( stocf * parameter("fftSize").toInt() / 2.);
  stocSize += stocSize % 2; // make it even for FFT-based resample function. (Essentia FFT algorithms only accepts even size).


 // resampling to decimate residual envelope
 std::vector<Real> magResDB;
 Real mag, magdB;

 for (int i=0; i< (int) fftRes.size(); i++)
  {

    mag =  sqrt( fftRes[i].real() * fftRes[i].real() +  fftRes[i].imag() * fftRes[i].imag());
    magdB = std::max(-200., 20. * log10( mag + 1e-10));
    magResDB.push_back(magdB);

   // _log << magdB << " ";
  }
 // _log << std::endl;

if (stocf == 1.){
  stocEnv = magResDB;
  std::cout << "do not resample stocenv. size= " << stocEnv.size() << std::endl;
}
else{
  // magResDB needs to be of even size to use resample with essentia FFT algorithms.
  if ((magResDB.size() % 2) > 0)
    magResDB.erase(magResDB.end()); // remove last  idx = (N/2) +1
  resample(magResDB, stocEnv, stocSize);
}



// resampled envelope
for (int i=0; i< (int) stocEnv.size(); i++) {
Real magInput =       sqrt( fftInput[i].real() * fftInput[i].real() +  fftInput[i].imag() * fftInput[i].imag());
    Real magInputdB = std::max(-200., 20. * log10( magInput + 1e-10));

Real magSine =       sqrt( fftSines[i].real() * fftSines[i].real() +  fftSines[i].imag() * fftSines[i].imag());
    Real magSinedB = std::max(-200., 20. * log10( magSine + 1e-10));

Real magRes =       sqrt( fftRes[i].real() * fftRes[i].real() +  fftRes[i].imag() * fftRes[i].imag());
    Real magResdB = std::max(-200., 20. * log10( magRes + 1e-10));

 //_log << magSinedB << " " << magResdB << " " << magInputdB << " ";
 _log << fftSines[i].real() << " " << fftSines[i].imag() << " " << fftRes[i].real() << " " << fftRes[i].imag() << " " << fftInput[i].real() << " " << fftInput[i].imag() << " ";
 //_log << stocEnv[i] << " " ;
}
_log << std::endl;

}



// Move this to new algorithm for ResampleFFT
void StochasticModelAnal::initializeFFT(std::vector<std::complex<Real> >&fft, int sizeFFT)
{
  fft.resize(sizeFFT);
  for (int i=0; i < sizeFFT; ++i){
    fft[i].real(0);
    fft[i].imag(0);
  }
}

*/

