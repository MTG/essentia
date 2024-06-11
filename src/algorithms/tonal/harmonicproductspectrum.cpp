/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#include "harmonicproductspectrum.h"
#include "essentiamath.h"
#include <complex>

using namespace std;
using namespace essentia;
using namespace standard;

const char* HarmonicProductSpectrum::name = "HarmonicProductSpectrum";
const char* HarmonicProductSpectrum::category = "Pitch";
const char* HarmonicProductSpectrum::description = DOC("This algorithm estimates the fundamental frequency given the spectrum of a monophonic music signal. It is an implementation of YinFFT algorithm [1], which is an optimized version of Yin algorithm for computation in the frequency domain. It is recommended to window the input spectrum with a Hann window. The raw spectrum can be computed with the Spectrum algorithm.\n"
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

void HarmonicProductSpectrum::configure() {
  // compute buffer sizes
  _frameSize = parameter("frameSize").toInt();
  _sampleRate = parameter("sampleRate").toReal();


  _tauMax = min(int(ceil(_sampleRate / parameter("minFrequency").toReal())), _frameSize/2);
  _tauMin = min(int(floor(_sampleRate / parameter("maxFrequency").toReal())), _frameSize/2);

  if (_tauMax <= _tauMin) {
    throw EssentiaException("HarmonicProductSpectrum: maxFrequency is lower than minFrequency, or they are too close, or they are out of the interval of detectable frequencies with respect to the specified frameSize. Minimum detectable frequency is ", _sampleRate / (_frameSize/2), " Hz");
  }

  // configure peak detection algorithm
  _peakDetect->configure("range", _frameSize/2+1,
                        "maxPeaks", 1,
                        "minPosition", _tauMin,
                        "maxPosition", _tauMax,
                        "orderBy", "amplitude");
}

void HarmonicProductSpectrum::compute() {
  const vector<Real>& spectrum = _spectrum.get();
  if (spectrum.empty()) {
    throw EssentiaException("HarmonicProductSpectrum: Cannot compute pitch detection on empty spectrum.");
  }
  Real& pitch = _pitch.get();
  Real& pitchConfidence = _pitchConfidence.get();

  if ((int)spectrum.size() != _frameSize/2+1) {//_sqrMag.size()/2+1) {
    Algorithm::configure( "frameSize", int(2*(spectrum.size()-1)) );
  }

  pitch = 0.0;
  pitchConfidence = 0.0;

}
