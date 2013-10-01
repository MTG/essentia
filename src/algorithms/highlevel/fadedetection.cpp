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

#include "fadedetection.h"
#include "essentiamath.h"

using namespace std;
using namespace TNT;

namespace essentia {
namespace standard {

const char* FadeDetection::name = "FadeDetection";
const char* FadeDetection::description = DOC("This algorithm computes two arrays containing the start/stop points of fade-ins and fade-outs detected in an audio file. The main hypothesis for the detection is that an increase or decrease of the RMS over time in an audio file corresponds to a fade-in or fade-out, repectively. Minimum and maximum mean-RMS-thresholds are used to define where fade-in and fade-outs occur.\n"
"\n"
"An exception is thrown if the input \"rms\" is empty.\n"
"\n"
"References:\n"
"  [1] Fade (audio engineering) - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Fade-in");

void FadeDetection::configure() {
  _frameRate = parameter("frameRate").toReal();
  _cutoffHigh = parameter("cutoffHigh").toReal();
  _cutoffLow = parameter("cutoffLow").toReal();
  _minLength = parameter("minLength").toReal();
}

void FadeDetection::compute() {

  const vector<Real>& rms = _rms.get();
  if (rms.empty()) {
    // throw exception as mean of empty arrays cannot be computed
    throw EssentiaException("FadeDetection: RMS array is empty");
  }
  Array2D<Real>& fade_in  = _fade_in.get();
  Array2D<Real>& fade_out = _fade_out.get();

  Real meanRms = mean(rms);
  Real thresholdHigh = _cutoffHigh * meanRms;
  Real thresholdLow = _cutoffLow * meanRms;
  int minLength = int(_minLength * _frameRate); // change minLength to samples

  // FADE-IN
  bool fade = false;
  vector<pair<int,int> > fade_in_vector;
  int fade_in_start = 0;
  int fade_in_stop;
  Real fade_in_start_value = 0.0;

  for (int i=0; i<int(rms.size()); ++i) {
    if (!fade) {
      // To get the fade-in start point
      if (rms[i] <= thresholdLow) {
        fade_in_start_value = rms[i];
        fade_in_start = i;
        fade = true;
      }
    }
    if (fade) {
      // To get the point with minimum energy as the fade-in starting point
      if (rms[i] < fade_in_start_value) {
        fade_in_start_value = rms[i];
        fade_in_start = i;
      }
      // To get the fade-in stop point
      if (rms[i] >= thresholdHigh) {
      	fade_in_stop = i;
        if ((fade_in_stop - fade_in_start) >= minLength) {
          fade_in_vector.push_back(make_pair(fade_in_start, fade_in_stop));
        }
        fade = false;
      }
    }
  }

  // convert units and push to output
  if (fade_in_vector.size() != 0) {
    fade_in = Array2D<Real>(int(fade_in_vector.size()), 2);
    for (int i=0; i<fade_in.dim1(); i++) {
      fade_in[i][0] = fade_in_vector[i].first / _frameRate;
      fade_in[i][1] = fade_in_vector[i].second / _frameRate;
    }
  }

  // FADE-OUT
  fade = false;
  vector<pair<int, int> > fade_out_vector;
  int fade_out_start;
  int fade_out_stop = 0;
  Real fade_out_stop_value = 0.0;

  for (int i=rms.size()-1; i>=0; i--) {
    if (!fade) {
      // To get the fade-out stop point
      if (rms[i] <= thresholdLow) {
        fade_out_stop_value = rms[i];
        fade_out_stop = i;
        fade = true;
      }
    }
    if (fade) {
      // To get the energy minimum for the fade-out stop point
      if (rms[i] <= fade_out_stop_value) {
        fade_out_stop_value = rms[i];
        fade_out_stop = i;
      }
      // To get the fade-out start point
      if (rms[i] >= thresholdHigh) {
      	fade_out_start = i;
        if ((fade_out_stop - fade_out_start) >= minLength) {
          fade_out_vector.push_back(make_pair(fade_out_start, fade_out_stop));
        }
      	fade = false;
      }
    }
  }

  // convert units and push to output
  if (fade_out_vector.size() != 0) {
    fade_out = Array2D<Real>(int(fade_out_vector.size()), 2);
    for (int i=0; i<fade_out.dim1(); i++) {
      fade_out[i][0] = fade_out_vector[fade_out_vector.size()-1-i].first / _frameRate;
      fade_out[i][1] = fade_out_vector[fade_out_vector.size()-1-i].second / _frameRate;
    }
  }
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

void FadeDetection::configure() {
  _fadeAlgo->configure("frameRate", parameter("frameRate").toReal(),
                       "cutoffHigh", parameter("cutoffHigh").toReal(),
                       "cutoffLow", parameter("cutoffLow").toReal(),
                       "minLength", parameter("minLength").toReal());
}

AlgorithmStatus FadeDetection::process() {
  while (_rms.acquire(1)) {
    _accu.push_back(*(Real*)_rms.getFirstToken());
    _rms.release(1);
  }

  if (!shouldStop()) return PASS;

  TNT::Array2D<Real> fadeIn, fadeOut;
  _fadeAlgo->input("rms").set(_accu);
  _fadeAlgo->output("fadeIn").set(fadeIn);
  _fadeAlgo->output("fadeOut").set(fadeOut);
  _fadeAlgo->compute();

  _fade_in.push(fadeIn);
  _fade_out.push(fadeOut);

  return OK;
}


void FadeDetection::reset () {
  Algorithm::reset();
  _fadeAlgo->reset();
  _accu.clear();
}

} // namespace streaming
} // namespace essentia
