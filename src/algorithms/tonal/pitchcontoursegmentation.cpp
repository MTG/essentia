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

#include "pitchcontoursegmentation.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PitchContourSegmentation::name = "PitchContourSegmentation";
const char* PitchContourSegmentation::category = "Pitch";
const char* PitchContourSegmentation::description = DOC("This algorithm converts a pitch sequence estimated from an audio signal into a set of discrete note events. Each note is defined by its onset time, duration and MIDI pitch value, quantized to the equal tempered scale.\n"
"\n"
"Note segmentation is performed based on pitch contour characteristics (island building) and signal RMS. Notes below an adjustable minimum duration are rejected.\n"
"\n"
"References:\n"
"  [1] R. J. McNab et al., \"Signal processing for melody transcription,\" in Proc. \n"
"  Proc. 19th Australasian Computer Science Conf., 1996");


void PitchContourSegmentation::configure() {
  _minDur = parameter("minDuration").toReal();
  _tuningFreq = parameter("tuningFrequency").toReal();
  _hopSize = parameter("hopSize").toReal();
  _sampleRate = parameter("sampleRate").toReal();
  _pitchDistanceThreshold = parameter("pitchDistanceThreshold").toReal();
  _rmsThreshold = parameter("rmsThreshold").toReal();

  _hopSizeFeat = 1024;
  _frameSizeFeat = 2048;
}

void PitchContourSegmentation::reSegment() {
    
  // find sequences of consecutive non-zero pitch values
  startC.clear();
  endC.clear();
  
  if (pitch[0] > 0) {
    startC.push_back(0);
  }
  for (int i = 0; i<(int)pitch.size()-1; i++) {
    if (pitch[i+1] > 0 && pitch[i] == 0) {
      startC.push_back(i+1);
    }
    if (pitch[i+1] == 0 && pitch[i] > 0) {
      endC.push_back(i);
    }
  }
  if (endC.size()<startC.size()) {
    endC.push_back(pitch.size()-1);
  }
}


void PitchContourSegmentation::compute() {
  
  // I/O
  const vector<Real>& pitchGlob = _pitch.get();
  const vector<Real>& signal = _signal.get();
  vector<Real>& onset = _onset.get();
  vector<Real>& duration = _duration.get();
  vector<Real>& MIDIpitch = _MIDIpitch.get();

  // we do not want to access the actual pitch contour -> create copy
  pitch = pitchGlob;
  
  // we assume a note onset at the beginning of a voiced section
  reSegment();
  
  // extract the RMS
  vector<Real> frame, rms;
  Real r;
  frameCutter = AlgorithmFactory::create("FrameCutter", "frameSize", _frameSizeFeat, "hopSize", _hopSizeFeat);
  RMS = AlgorithmFactory::create("RMS");
  frameCutter->input("signal").set(signal);
  frameCutter->output("frame").set(frame);
  RMS->input("array").set(frame);
  RMS->output("rms").set(r);
  while (true){
    frameCutter->compute();
    if (!frame.size()) {
      break;
    }
    RMS->compute();
    rms.push_back(r);
  }
  delete frameCutter;
  delete RMS;

  // segment based on pitch distance ("island building")
  minDurPitchSamples = round(_minDur * Real(_sampleRate)/Real(_hopSize));
  for (int i=0; i<(int)startC.size(); i++) {
    if (endC[i]-startC[i] > 2*minDurPitchSamples) {
      vector<Real> contourH, contourC;
      // contour in Hz
      for (int ii = startC[i]; ii <= endC[i]; ii++){
        contourH.push_back(pitch[ii]);
      }
      // contour in cents
      for (int ii = 0; ii <= (int)contourH.size(); ii++){
        contourC.push_back(1200 * log2( contourH[ii] / _tuningFreq ));
      }
      // running mean
      Real av = mean(contourC, 0, minDurPitchSamples);
      int j = minDurPitchSamples + 1; // running index
      int k = 0; // current note start
      while (j < (int)contourC.size() - minDurPitchSamples){
        if ( abs( contourC[j] - av )< _pitchDistanceThreshold ) {
          av = mean( contourC, k, j );
          j++;
        }
        else{
          pitch[startC[i] + j] = 0;
          k = j;
          j = j + minDurPitchSamples;
          av = mean(contourC, k, j);
        }
      }
    }
  }
 
  reSegment();

  // segment based on rms
  Real resampleFactor = _hopSizeFeat / _hopSize;
  for (int i=0; i<(int)startC.size(); i++) {
    if (endC[i]-startC[i] > 2*minDurPitchSamples) {
      vector<Real> rmsSeg;
      for (int ii=startC[i]; ii<=endC[i]; ii++) {
        rmsSeg.push_back(rms[round(Real(ii) / resampleFactor)]);
      }
      Real m = mean(rmsSeg, 0, rmsSeg.size()-1);
      Real s = stddev(rmsSeg, m);
      int ii = minDurPitchSamples;
      while (ii < (int)rmsSeg.size() - minDurPitchSamples) {
        Real zs = (rmsSeg[ii] - m) / s;
        if (zs < _rmsThreshold){
          pitch[startC[i] + ii] = 0;
          ii = ii + minDurPitchSamples;
        }
        else{
          ii++;
        }
      }
    }
  }
    
  reSegment();
    
  // assign pitch values
  for (int i=0; i<(int)startC.size(); i++) {
    vector<Real> contour;
    onset.push_back(Real(startC[i]) * Real(_hopSize) / Real(_sampleRate));
    Real d = endC[i] - startC[i];
    duration.push_back(d * Real(_hopSize) / Real(_sampleRate));
    Real meanFreq=mean(pitch,startC[i], endC[i]-1);
    MIDIpitch.push_back(round(12*log2(meanFreq/440) + 69));
  } 
}

