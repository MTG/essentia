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

#include "pitchyin.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;


const char* PitchYin::name = "PitchYin";
const char* PitchYin::category = "Pitch";
const char* PitchYin::description = DOC("This algorithm estimates the fundamental frequency given the frame of a monophonic music signal. It is an implementation of the Yin algorithm [1] for computations in the time domain.\n"
"\n"
"An exception is thrown if an empty signal is provided.\n"
"\n"
"Please note that if \"pitchConfidence\" is zero, \"pitch\" is undefined and should not be used for other algorithms.\n"
"Also note that a null \"pitch\" is never ouput by the algorithm and that \"pitchConfidence\" must always be checked out.\n"
"\n"
"References:\n"
"  [1] De Cheveigné, A., & Kawahara, H. \"YIN, a fundamental frequency estimator\n"
"  for speech and music. The Journal of the Acoustical Society of America,\n"
"  111(4), 1917-1930, 2002.\n\n"
"  [2] Pitch detection algorithm - Wikipedia, the free encyclopedia\n"
"  http://en.wikipedia.org/wiki/Pitch_detection_algorithm");

void PitchYin::configure() {
  _frameSize = parameter("frameSize").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _tolerance = parameter("tolerance").toReal();
  _interpolate = parameter("interpolate").toBool();

  _yin.resize(_frameSize/2 + 1);

  _tauMax = min(int(ceil(_sampleRate / parameter("minFrequency").toReal())), _frameSize/2);
  _tauMin = min(int(floor(_sampleRate / parameter("maxFrequency").toReal())), _frameSize/2);

  if (_tauMax <= _tauMin) {
    throw EssentiaException("PitchYin: maxFrequency is lower than minFrequency, or they are too close, or they are out of the interval of detectable frequencies with respect to the specified frameSize.");
  }

  _peakDetectLocal->configure("interpolate", _interpolate,
                              "range", _frameSize/2+1,
                              "maxPeaks", 1,
                              "minPosition", _tauMin,
                              "maxPosition", _tauMax,
                              "orderBy", "position", 
                              "threshold", -1 * _tolerance);

  _peakDetectGlobal->configure("interpolate", _interpolate,
                              "range", _frameSize/2+1,
                              "maxPeaks", 1,
                              "minPosition", _tauMin,
                              "maxPosition", _tauMax,
                              "orderBy", "amplitude");
}


void PitchYin::compute() {
  const vector<Real>& signal = _signal.get();
  if (signal.empty()) {
    throw EssentiaException("PitchYin: Cannot compute pitch detection on empty signal frame.");
  }
  if ((int) signal.size() != _frameSize) {
    Algorithm::configure( "frameSize", int(signal.size()) );
  } 

  Real& pitch = _pitch.get();
  Real& pitchConfidence = _pitchConfidence.get();
  
  _yin[0] = 1.;

  // Compute difference function
  for (int tau=1; tau < (int) _yin.size(); ++tau) {
    _yin[tau] = 0.;
    for (int j=0; j < (int) _yin.size()-1; ++j) {
      _yin[tau] += pow(signal[j] - signal[j+tau], 2);
    }
  }

  // Compute a cumulative mean normalized difference function
  Real sum = 0.; 
  for (int tau=1; tau < (int) _yin.size(); ++tau) {
    sum += _yin[tau];
    _yin[tau] = _yin[tau] * tau / sum;

    // Cannot simply check for sum==0 because NaN will be also produced by 
    // infinitely small values 
    if (isnan(_yin[tau])) {
      _yin[tau] = 1;
    }
  }

  // _yin[tau] is equal to 1 in the case if the df value for 
  // this tau is the same as the mean across all df values from 1 to tau

  // Detect best period
  Real period = 0.;
  Real yinMin = 0.;

  // Select the local minima below the absolute threshold with the smallest 
  // period value. Use global minimum if no minimas were found below threshold.

  // invert yin values because we want to detect the minima while PeakDetection
  // detects the maxima
 
  for(int tau=0; tau < (int) _yin.size(); ++tau) {
    _yin[tau] = -_yin[tau];
  }

  _peakDetectLocal->input("array").set(_yin);
  _peakDetectLocal->output("positions").set(_positions);
  _peakDetectLocal->output("amplitudes").set(_amplitudes);
  _peakDetectLocal->compute();    
   
  if (_positions.size()) {
    period = _positions[0];
    yinMin = -_amplitudes[0];
  }
  else {
    // no minima found below the threshold --> find the global minima
    _peakDetectGlobal->input("array").set(_yin);
    _peakDetectGlobal->output("positions").set(_positions);
    _peakDetectGlobal->output("amplitudes").set(_amplitudes);
    _peakDetectGlobal->compute();    

    if (_positions.size()) {
      period = _positions[0];
      yinMin = -_amplitudes[0];
    }
  }

  // NOTE: not sure if it is faster to run peak detection once to detect many 
  // peaks and process the values manually instead of running peak detection 
  // twice and detecting only two peaks. 


  // Treat minimas with yin values >= 1 and respective negative confidence values 
  // as completely unreliable (in the case if they occur in results). 
  
  if (period) {
    pitch = _sampleRate / period;
    pitchConfidence = 1. - yinMin;
    if (pitchConfidence < 0) {
      pitchConfidence = 0.;
    }
  }
  else {
    pitch = 0.;
    pitchConfidence = 0.;
  }      
}
