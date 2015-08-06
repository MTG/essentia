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

#include "sinemodelanal.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* SineModelAnal::name = "SineModelAnal";
const char* SineModelAnal::description = DOC("This algorithm computes the sine model analysis without sine tracking. \n"
"\n"
"It is recommended that the input \"spectrum\" be computed by the Spectrum algorithm. This algorithm uses PeakDetection. See documentation for possible exceptions and input requirements on input \"spectrum\".\n"
"\n"
"References:\n"
"  [1] Peak Detection,\n"
"  http://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html");


// TODO: process multiple frames at the same time to do tracking in standard mode.
// Initial implementation witohut tracking for both standard and streaming mode.
// Next step: implement sine tracking for standard implementation, if we have access to all spectrogram.

void SineModelAnal::configure() {

  std::string orderBy = parameter("orderBy").toLower();
  if (orderBy == "magnitude") {
    orderBy = "amplitude";
  }
  else if (orderBy == "frequency") {
    orderBy = "position";
  }
  else {
    throw EssentiaException("Unsupported ordering type: '" + orderBy + "'");
  }

  _peakDetect->configure("interpolate", true,
                         "range", parameter("sampleRate").toReal()/2.0,
                         "maxPeaks", parameter("maxPeaks"),
                         "minPosition", parameter("minFrequency"),
                         "maxPosition", parameter("maxFrequency"),
                         "threshold", parameter("magnitudeThreshold"),
                         "orderBy", orderBy);

}



void SineModelAnal::sinusoidalTracking(std::vector<Real>& peakMags, std::vector<Real>& peakFrequencies, std::vector<Real>& peakPhases, const std::vector<Real> tfreq, Real freqDevOffset=20, Real freqDevSlope=0.01){
//	pfreq, pmag, pphase: frequencies and magnitude of current frame
//	tfreq: frequencies of incoming tracks from previous frame
//	freqDevOffset: minimum frequency deviation at 0Hz
//	freqDevSlope: slope increase of minimum frequency deviation

  freqDevOffset=20;
  freqDevSlope=0.01;


 // TODO: if multiple frames as input are given. Only supported in standard mode.


}

void SineModelAnal::phaseInterpolation(std::vector<Real> fftphase, std::vector<Real> peakFrequencies, std::vector<Real>& peakPhases){

  int N = peakFrequencies.size();
  peakPhases.resize(N);

  int idx;
  float  a, pos;
  int fftSize = fftphase.size();

  for (int i=0; i < N; ++i){
    // linear interpolation. (as done in numpy.interp function)
    pos =  fftSize * (peakFrequencies[i] / (parameter("sampleRate").toReal()/2.0) );
    idx = int ( 0.5 + pos ); // closest index

    a = pos - idx; // interpolate factor
    // phase diff smaller than PI to do intperolation and avoid jumps
    if (a < 0 && idx > 0){
      peakPhases[i] =  (abs(fftphase[idx-1] - fftphase[idx]) > Real(M_PI)) ? a * fftphase[idx-1] + (1.0 -a) * fftphase[idx] : fftphase[idx];
    }
    else {
      if (idx < fftSize-1 ){
      peakPhases[i] = (abs(fftphase[idx+1] - fftphase[idx]) > Real(M_PI)) ? a * fftphase[idx+1] + (1.0 -a) * fftphase[idx]: fftphase[idx];
      }
      else {
       peakPhases[i] = fftphase[idx];
     }
    }
  }
}


void SineModelAnal::compute() {

  const std::vector<std::complex<Real> >& fft = _fft.get();
  std::vector<Real>& peakMagnitude = _magnitudes.get();
  std::vector<Real>& peakFrequency = _frequencies.get();
  std::vector<Real>& peakPhase = _phases.get();

  std::vector<Real> fftmag;
  std::vector<Real> fftphase;

  _cartesianToPolar->input("complex").set(fft);
  _cartesianToPolar->output("magnitude").set(fftmag);
  _cartesianToPolar->output("phase").set(fftphase);
  _peakDetect->input("array").set(fftmag);
  _peakDetect->output("positions").set(peakFrequency);
  _peakDetect->output("amplitudes").set(peakMagnitude);

  _cartesianToPolar->compute();
  _peakDetect->compute();

  phaseInterpolation(fftphase, peakFrequency, peakPhase);
}


