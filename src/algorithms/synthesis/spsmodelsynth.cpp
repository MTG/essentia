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

#include "spsmodelsynth.h"
#include "essentiamath.h"


using namespace essentia;
using namespace standard;


const char* SpsModelSynth::name = "SpsModelSynth";
const char* SpsModelSynth::description = DOC("This algorithm computes the sinusoidal plus stochastic model synthesis from SPS model analysis.");



// configure
void SpsModelSynth::configure()
{
  _sampleRate = parameter("sampleRate").toReal();
  _fftSize = parameter("fftSize").toInt();
  _hopSize = parameter("hopSize").toInt();

  _sineModelSynth->configure( "sampleRate", _sampleRate,
                            "fftSize", _fftSize,
                            "hopSize", _hopSize
                            );

 _stochasticModelSynth->configure("fftSize", 2* parameter("hopSize").toInt(),
                            "hopSize", parameter("hopSize").toInt(),
                            "stocf", parameter("stocf").toReal());


  _ifftSine->configure("size", _fftSize);

  _overlapAdd->configure( "frameSize", _fftSize, // uses synthesis window
													"hopSize", _hopSize);


_log.open("synth.log");
}


void SpsModelSynth::compute() {

  const std::vector<Real>& magnitudes = _magnitudes.get();
  const std::vector<Real>& frequencies = _frequencies.get();
  const std::vector<Real>& phases = _phases.get();
  const std::vector<Real>& stocenv = _stocenv.get();

  //std::vector<std::complex<Real> >& outfft = _outfft.get();
  std::vector<Real>& outframe = _outframe.get();


  // temp vectors
  std::vector<std::complex<Real> > fftSines;
  std::vector<std::complex<Real> > fftStoc;
  std::vector<Real> wsineFrame; // windowed frames
  std::vector<Real> sineFrame;  // overlap output frame
  std::vector<Real> stocFrame;  // output stochastic frame

  int i = 0;

  _sineModelSynth->input("magnitudes").set(magnitudes);
  _sineModelSynth->input("frequencies").set(frequencies);
  _sineModelSynth->input("phases").set(phases);
  _sineModelSynth->output("fft").set(fftSines);

  _sineModelSynth->compute();

  //std::vector<Real> sineAudio, resAudio;
  _ifftSine->input("fft").set(fftSines);
  _ifftSine->output("frame").set(wsineFrame);
  _ifftSine->compute();
  _overlapAdd->input("signal").set(wsineFrame);
  _overlapAdd->output("signal").set(sineFrame);
  _overlapAdd->compute();

// TODO: implement
// synthesis of the stochastic component
  _stochasticModelSynth->input("stocenv").set(stocenv);
  _stochasticModelSynth->output("frame").set(stocFrame);
  _stochasticModelSynth->compute();

printf("ifftout:%d, olap:%d, stoc:%d \t", wsineFrame.size(), sineFrame.size(), stocFrame.size());
// add sine and stochastic copmponents
 for (i = 0; i < (int)stocFrame.size(); ++i)
  {
    outframe.push_back(sineFrame[i] + 0*stocFrame[i]);
    outframe.push_back(sineFrame[i] + 0*stocFrame[i]);
  }


/* OLD code
  // stochastic
  int outSize =  (int)floor(_fftSize/2.0) + 1;
  initializeFFT(outfft, outSize);

  //# synthesize stochastic residual
  stochasticModelSynthOLD(stocenv, parameter("hopSize").toInt(), outSize, fftStoc);

  // mix stoachastic and sinusoidal components
  for (i = 0; i < (int)outfft.size(); ++i)
  {

     outfft[i].real(0*fftSines[i].real() + fftStoc[i].real());
     outfft[i].imag( 0*fftSines[i].imag() + fftStoc[i].imag());
  }
  */

// DEBUG
// output is an audio frame / already overlapp-add. Directly to write inot output buffer.
//outframe.resize(parameter("hopSize").toInt());
//std::fill(outframe.begin(), outframe.end(), 0.);

}

/*
void SpsModelSynth::stochasticModelSynthOLD(const std::vector<Real> stocEnv, const int H, const int N, std::vector<std::complex<Real> > &fftStoc)
{
//	"""
//	Stochastic synthesis of a sound
//	stocEnv: stochastic envelope; H: hop size; N: fft size (postive??)
//	returns y: output FFT magitude
//	"""

  fftStoc.resize(N);  // init stochastic FFT

  Real magdB;
  Real phase;
  std::vector<Real> stocEnv2;
  std::vector<Real> stocEnvOut;

  // copy last value to avoid discotninuities at Nyquist frequency
  if (parameter("stocf").toReal() == 1.)
  {
    stocEnvOut = stocEnv;
    std::cout << "synth debug: copy do not resample: stocenv size= " << stocEnvOut.size() <<std::endl;
  }
  else
  {
  stocEnv2 = stocEnv;
  while (_stocSize > (int)stocEnv2.size()){
    stocEnv2.push_back(stocEnv2[stocEnv2.size()-1]);
    }
  resample(stocEnv2, stocEnvOut, N);   // resampling will produce a eve-sized vector due to itnernal FFT algorithm
 }

  // copy last value if size differ
  while (N > (int)stocEnvOut.size())
    stocEnvOut.push_back(stocEnvOut[stocEnvOut.size()-1]);

for (int i = 0; i < (int)stocEnvOut.size(); ++i){
_log  <<stocEnvOut[i] << " ";
}

  for (int i = 0; i < N; ++i)
  {
    phase =  2 * M_PI *  Real(rand()/Real(RAND_MAX));
    magdB = stocEnvOut[i];
//if (i > N/2)
//  magdB = -200;
    // positive spectrums
    fftStoc[i].real( powf(10.f, (magdB / 20.f)) * cos(phase) ) ;
    fftStoc[i].imag( powf(10.f, (magdB / 20.f)) * sin(phase) ) ;
  }
  _log << std::endl;
}


void SpsModelSynth::initializeFFT(std::vector<std::complex<Real> >&fft, int sizeFFT)
{
  fft.resize(sizeFFT);
  for (int i=0; i < sizeFFT; ++i){
    fft[i].real(0);
    fft[i].imag(0);
  }
}

// function to resample based on the FFT
// Use the same function than in python code
// http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html
void SpsModelSynth::resample(const std::vector<Real> in, std::vector<Real> &out, const int sizeOut)
{

// TODO: consider adding this algorithhms as an essentia standard algorithm

  std::vector<std::complex<Real> >fftin; // temp vectors
  std::vector<std::complex<Real> >fftout; // temp vectors
  std::vector<Real> ifftout; // temp vectors

  int sizeIn = (int) in.size();

  _fft->input("frame").set(in);
  _fft->output("fft").set(fftin);
  _fft->compute();


  int hN = (sizeIn/2.)+1;
  int hNout = (_stocSpecSize/2.)+1;; // (sizeOut/2.)+1;
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

//std::cout << "res out: " <<ifftout.size() << " _stocSpecSize: " << _stocSpecSize << std::endl;
  // normalize
  Real normalizationGain = 1. / float(sizeIn);
  for (int i = 0; i < sizeOut; ++i)
  {
    out.push_back(ifftout[i] * normalizationGain) ;
  }

}

*/
