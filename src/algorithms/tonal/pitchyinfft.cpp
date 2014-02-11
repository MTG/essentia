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

#include "pitchyinfft.h"
#include "essentiamath.h"
#include <complex>

using namespace std;
using namespace essentia;
using namespace standard;

static const Real _freqsMask[] = {0., 20., 25., 31.5, 40., 50., 63., 80.,
  100., 125., 160., 200., 250., 315., 400., 500., 630., 800., 1000., 1250.,
	1600., 2000., 2500., 3150., 4000., 5000., 6300., 8000., 9000., 10000.,
	12500., 15000., 20000.,  25100};

static const Real _weightMask[] = {-75.8, -70.1, -60.8, -52.1, -44.2, -37.5,
	-31.3, -25.6, -20.9, -16.5, -12.6, -9.6, -7.0, -4.7, -3.0, -1.8, -0.8,
	-0.2, -0.0, 0.5, 1.6, 3.2, 5.4, 7.8, 8.1, 5.3, -2.4, -11.1, -12.8,
	-12.2, -7.4, -17.8, -17.8, -17.8};

const char* PitchYinFFT::name = "PitchYinFFT";
const char* PitchYinFFT::description = DOC("This algorithm estimates the fundamental frequency from a given spectrum. It is an implementation of YinFFT algorithm [1], which is an optimizated version of Yin algorithm for computations in the frequency domain. It is recommended to window the input spectrum with a Hann window. The raw spectrum can be computed with the Spectrum algorithm.\n"
"\n"
"An exception is thrown if an empty spectrum is provided.\n"
"\n"
"Please note that if \"pitchConfidence\" is zero, \"pitch\" is undefined and should not be used for other algorithms.\n"
"Also note that a null \"pitch\" is never ouput by the algorithm and that \"pitchConfidence\" must always be checked out.\n"
"\n"
"References:\n"
"  [1] P. M. Brossier, \"Automatic Annotation of Musical Audio for Interactive\n"
"  Applications,” QMUL, London, UK, 2007.\n\n"
"  [2] Pitch detection algorithm - Wikipedia, the free encyclopedia\n"
"  http://en.wikipedia.org/wiki/Pitch_detection_algorithm");

void PitchYinFFT::configure() {
  // compute buffer sizes
  _frameSize = parameter("frameSize").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _interpolate = parameter("interpolate").toBool();
  _sqrMag.resize(_frameSize);
  _weight.resize(_frameSize/2+1);
  _yin.resize(_frameSize/2+1);
  // configure algorithms
  _fft->configure("size", _frameSize);
  // allocate memory
  spectralWeights();

  _tauMax = min(int(ceil(_sampleRate / parameter("minFrequency").toReal())), _frameSize/2);
  _tauMin = min(int(floor(_sampleRate / parameter("maxFrequency").toReal())), _frameSize/2);

  if (_tauMax <= _tauMin) {
    throw EssentiaException("PitchYinFFT: maxFrequency is lower than minFrequency, or they are too close, or they are out of the interval of detectable frequencies with respect to the specified frameSize. Minimum detectable frequency is ", _sampleRate / (_frameSize/2), " Hz");
  }

  // configure peak detection algorithm
  _peakDetect->configure("interpolate", _interpolate,
                        "range", _frameSize/2+1,
                        "maxPeaks", 1,
                        "minPosition", _tauMin,
                        "maxPosition", _tauMax,
                        "orderBy", "amplitude");
}

void PitchYinFFT::spectralWeights() {
  int i = 0, j = 1;
  Real freq = 0, a0 = 0, a1 = 0, f0 = 0, f1 = 0;
  for (i=0; i < int(_weight.size()); ++i) {
    freq = (Real)i/(Real)_frameSize*_sampleRate;
    while (freq > _freqsMask[j]) {
      j +=1;
    }
    a0 = _weightMask[j-1];
    f0 = _freqsMask[j-1];
    a1 = _weightMask[j];
    f1 = _freqsMask[j];
    if (f0 == f1) { // just in case
      _weight[i] = a0;
    }
    else if (f0 == 0) { // y = ax+b
      _weight[i] = (a1-a0)/f1*freq + a0;
    }
    else {
      _weight[i] = (a1-a0)/(f1-f0)*freq +
        (a0 - (a1 - a0)/(f1/f0 - 1.));
    }
    while (freq > _freqsMask[j]) {
      j +=1;
    }
    // could be sqrt of this too
    _weight[i] = db2lin(_weight[i]/2.0);
  }
}

void PitchYinFFT::compute() {
  const vector<Real>& spectrum = _spectrum.get();
  if (spectrum.empty()) {
    throw EssentiaException("PitchYinFFT: Cannot compute pitch detection on empty spectrum.");
  }
  Real& pitch = _pitch.get();
  Real& pitchConfidence = _pitchConfidence.get();
  int l = 0;
  Real yinMin, tau, tmp = 0, sum = 0;

  if ((int)spectrum.size() != _frameSize/2+1) {//_sqrMag.size()/2+1) {
    Algorithm::configure( "frameSize", int(2*(spectrum.size()-1)) );
  }

  // build modified squared difference function using a weighted
  // input norm spectrum
  vector<complex<Real> > frameFFT;
  _fft->input("frame").set(_sqrMag);
  _fft->output("fft").set(frameFFT);

  // used to get phase and norm
  _cart2polar->input("complex").set(frameFFT);
  _cart2polar->output("magnitude").set(_resNorm);
  _cart2polar->output("phase").set(_resPhase);

  _sqrMag[0] = spectrum[0]*spectrum[0]*_weight[0];
  sum += _sqrMag[0];
  for (l=1; l < (int)spectrum.size(); l++) {
    _sqrMag[l] = spectrum[l]*spectrum[l]*_weight[l];
    _sqrMag[_frameSize-l] = _sqrMag[l];
    sum += _sqrMag[l];
  }
  sum *= 2;

  if (sum==0) {
    // a frame with silence or too quiet, cannot predict pitch
    // as the division by tmp will produce NaN
    pitch = 0.0;
    pitchConfidence = 0.0;
    return;
  }

  _fft->compute();
  _cart2polar->compute();
  _yin[0] = 1.;
  for (tau = 1; tau < int(_yin.size()); ++tau) {
    _yin[tau] = sum - _resNorm[tau]*cos(_resPhase[tau]);
    tmp += _yin[tau];
    _yin[tau] *= tau/tmp;
  }

  // search for argmin within minTau/maxTau range
  if (_interpolate) {
    // yin values are in the range [0,inf], because we want to detect the minima and peak detection detects the maxima,
    // yin values will be inverted
    for(int n=0; n<=_yin.size(); ++n) {
      _yin[n] = -_yin[n];
    }
    // use interal peak detection algorithm
    _peakDetect->input("array").set(_yin);
    _peakDetect->output("positions").set(_positions);
    _peakDetect->output("amplitudes").set(_amplitudes);
    _peakDetect->compute();    
    tau = _positions[0];
    yinMin = -_amplitudes[0];
  }
  else {
    // with no interpolation is faster to simply search the minimum
    int int_tau = _tauMin;
    yinMin = _yin[_tauMin];
    for (int i=_tauMin; i<=_tauMax; ++i) {
      if (_yin[i] < yinMin) {
        int_tau = i;
        yinMin = _yin[i];
      }
    }
    tau = int_tau + 0.;
  }

  if (tau != 0.0) {
    pitch = _sampleRate / tau;
    pitchConfidence = 1. - yinMin;
  }
  else {
    pitch = 0.0;
    pitchConfidence = 0.0;
  }      

}
