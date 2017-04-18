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

#include "vibrato.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Vibrato::name = "Vibrato";
const char* Vibrato::category = "Pitch";
const char* Vibrato::description = DOC("This algorithm detects the presence of vibrato and estimates its parameters given a pitch contour [Hz]. The result is the vibrato frequency in Hz and the extent (peak to peak) in cents. If no vibrato is detected in a frame, the output of both values is zero.\n"
"\n"
"This algorithm should be given the outputs of a pitch estimator, i.e. PredominantMelody, PitchYinFFT or PitchMelodia and the corresponding sample rate with which it was computed.\n"
"\n"
"The algorithm is an extended version of the vocal vibrato detection in PerdominantMelody."
"\n"
"References:\n"
"  [1] J. Salamon and E. Gómez, \"Melody extraction from polyphonic music\n"
"  signals using pitch contour characteristics,\" IEEE Transactions on Audio,\n"
"  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.\n\n");


void Vibrato::configure() {
    
  _minFrequency = parameter("minFrequency").toReal();
  _maxFrequency = parameter("maxFrequency").toReal();
  _minExtend = parameter("minExtend").toReal();
  _maxExtend = parameter("maxExtend").toReal();
  _sampleRate = parameter("sampleRate").toReal();
    
  frameSize = int(0.350 * _sampleRate);
  fftSize = 4*frameSize;
  
  frameCutter->configure("frameSize", frameSize, "hopSize", 1, "startFromZero", true);
  window->configure("type", "hann", "zeroPadding", 3*frameSize);
  spectrum->configure("size", fftSize);
  spectralPeaks->configure("sampleRate", _sampleRate, "maxPeaks", 3, "orderBy", "magnitude");
}

void Vibrato::compute() {
  
  Real vibdBDropLobe = 15.;
  Real vibdBDropSecondPeak = 20.;
    
  const vector<Real>& pitch = _pitch.get();
  vector<Real>& vibratoFrequency =_vibratoFrequency.get();
  vector<Real>& vibratoExtend = _vibratoExtend.get();

  // if pitch vector is empty
  if (pitch.empty()) {
    vibratoFrequency.clear();
    vibratoExtend.clear();
    return;
  }

  vibratoFrequency.assign(pitch.size(), 0.);
  vibratoExtend.assign(pitch.size(), 0.);
  
  vector<Real> pitchP;
    
  // set negative pitch values to zero
  for (int i=0; i<(int)pitch.size(); i++) {
    if (pitch[i]<0) {
        pitchP.push_back(0.0);
    } else {
        pitchP.push_back(pitch[i]);
    }
  }

  // get contour start and end indices
  vector<Real> startC, endC;
  if (pitchP[0]>0){
    startC.push_back(0);
  }
  for (int i=0; i<(int)pitchP.size()-1; i++) {
    if (pitchP[i+1]>0 && pitchP[i]==0) {
      startC.push_back(i+1);
    }
    if (pitchP[i+1]==0 && pitchP[i]>0) {
      endC.push_back(i);
    }
  }
  if (endC.size()<startC.size()) {
    endC.push_back(pitch.size()-1);
  }

  // iterate over contour segments
  for (int i=0; i<(int)startC.size(); i++) {
    // get a segment in cents
    vector<Real> contour;
    for (int ii=startC[i]; ii<=endC[i]; ii++) {
      contour.push_back(1200*log2(pitch[ii]/55.0));
    }
      
    // setup algorithm I/O
    vector<Real> frame;
    frameCutter->input("signal").set(contour);
    frameCutter->output("frame").set(frame);
    vector<Real> windowedFrame;
    window->input("frame").set(frame);
    window->output("frame").set(windowedFrame);
    vector<Real> vibSpectrum;
    spectrum->input("frame").set(windowedFrame);
    spectrum->output("spectrum").set(vibSpectrum);
    vector<Real> peakFrequencies, peakMagnitudes;
    spectralPeaks->input("spectrum").set(vibSpectrum);
    spectralPeaks->output("frequencies").set(peakFrequencies);
    spectralPeaks->output("magnitudes").set(peakMagnitudes);
    frameCutter->reset();
    
    int frameNumber=0;
  
    // frame-wise processing
    while (true) {
          
      //get a frame
      frameCutter->compute();
      frameNumber++;
 
      if(!frame.size()) {
        break;
      }

      // subtract mean pitch from frame
      Real m = mean(frame, 0, frame.size()-1);
      for (int ii=0; ii<(int)frame.size(); ii++) {
        frame[ii]-=m;
      }
          
      // spectral peaks
      window->compute();
      spectrum->compute();
      spectralPeaks->compute();
          
      int numberPeaks = peakFrequencies.size();
      if (!numberPeaks) {
        continue;
      }
          
      if (peakFrequencies[0] < _minFrequency || peakFrequencies[0] > _maxFrequency) {
        continue;
      }

      if (numberPeaks > 1) {  // there is at least one extra peak
        if (peakFrequencies[1] <= _maxFrequency) {
          continue;
        }
        if (20 * log10(peakMagnitudes[0]/peakMagnitudes[1]) < vibdBDropLobe) {
          continue;
        }
      }
          
      if (numberPeaks > 2) {  // there is a second extra peak
        if (peakFrequencies[2] <= _maxFrequency) {
          continue;
        }
        if (20 * log10(peakMagnitudes[0]/peakMagnitudes[2]) < vibdBDropSecondPeak) {
          continue;
        }
      }
      
      Real ext = frame[argmax(frame)] + abs(frame[argmin(frame)]);
      if (ext<_minExtend || ext>_maxExtend){
        continue;
      }

      int ii = startC[i]+frameNumber-1;
      vibratoFrequency[ii] = peakFrequencies[0];
      vibratoExtend[ii] = ext;
      // NOTE: no need to loop over the frame, as the hopSize is 1
      /*
      for (int ii=startC[i]+frameNumber-1; ii<startC[i]+frameNumber-1+frameSize; ii++) {  
        vibratoFrequency[ii]=peakFrequencies[0];
        vibratoExtend[ii]=ext;
      }
      */
    }
  }
}

Vibrato::~Vibrato() {
  delete frameCutter;
  delete window;
  delete spectrum;
  delete spectralPeaks;
}

void Vibrato::reset() {
  Algorithm::reset();
  frameCutter->reset();
  spectralPeaks->reset();
  spectrum->reset();
  window->reset();
}
