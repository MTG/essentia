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
const char* HarmonicProductSpectrum::description = DOC("This algorithm estimates the fundamental frequency given the spectrum of a monophonic music signal. It is an implementation of Harmonic Product Spectrum algorithm [1], computed in the frequency-domain. It is recommended to window the input spectrum with a Hann window. The raw spectrum can be computed with the Spectrum algorithm.\n"
"\n"
"An exception is thrown if an empty spectrum is provided.\n"
"\n"
"Note that a null “pitch” is never output by the algorithm.\n"
"\n"
"References:\n"
"  [1] Noll, A. M. (1970). Pitch Determination of Human Speech by the\n"
"  Harmonic Product Spectrum, the Harmonic Sum Spectrum, and a Maximum\n"
"  Likelihood Estimate. Symposium on Computer Processing in Communication,\n"
"  Ed., 19, 779–797.");

void HarmonicProductSpectrum::configure() {
  // compute buffer sizes
  _frameSize = parameter("frameSize").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _numHarmonics = parameter("numHarmonics").toInt();
  _magnitudeThreshold = parameter("magnitudeThreshold").toReal();


  _tauMax = min(int(ceil(_sampleRate / parameter("minFrequency").toReal())), _frameSize/2);
  _tauMin = min(int(floor(_sampleRate / parameter("maxFrequency").toReal())), _frameSize/2);

  if (_tauMax <= _tauMin) {
    throw EssentiaException("HarmonicProductSpectrum: maxFrequency is lower than minFrequency, or they are too close, or they are out of the interval of detectable frequencies with respect to the specified frameSize. Minimum detectable frequency is ", _sampleRate / (_frameSize/2), " Hz");
  }

  // configure peak detection algorithm
  _peakDetect->configure("range", _frameSize/2+1,
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

  vector<Real> hps(spectrum);

  for (int h=2; h < _numHarmonics; h++) {

      vector<Real> downsampled(spectrum.size()/h, 1.0);
      for (int bin=0; bin < downsampled.size(); bin++) {
          downsampled[bin] = spectrum[bin*h];
      }
      for (int bin=0; bin < downsampled.size(); bin++) {
          hps[bin] *= downsampled[bin];
      }
  }

  for (int bin=hps.size()/_numHarmonics; bin < hps.size(); bin++) {
      hps[bin] = 0;
  }
  vector<Real> _positions;
  vector<Real> _amplitudes;

  _peakDetect->input("array").set(hps);
  _peakDetect->output("positions").set(_positions);
  _peakDetect->output("amplitudes").set(_amplitudes);
  _peakDetect->compute();

  if (_positions.size() == 0) {
    pitch = 0.0;
    pitchConfidence = 0.0;
  } else {
    pitch = _positions[0] * _sampleRate / _frameSize;
    pitchConfidence = 1.0;
  }
}
