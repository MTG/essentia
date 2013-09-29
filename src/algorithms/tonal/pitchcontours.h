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

#ifndef ESSENTIA_PITCHCONTOURS_H
#define ESSENTIA_PITCHCONTOURS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class PitchContours : public Algorithm {

 private:
  Input<std::vector<std::vector<Real> > > _peakBins;
  Input<std::vector<std::vector<Real> > > _peakSaliences;

  Output<std::vector<std::vector<Real> > > _contoursBins;
  Output<std::vector<std::vector<Real> > > _contoursSaliences;
  Output<std::vector<Real> > _contoursStartTimes;
  Output<Real> _duration;

  Real _sampleRate;
  int _hopSize;
  Real _binResolution;
  Real _peakFrameThreshold;
  Real _peakDistributionThreshold;
  //Real _timeContinuity;

  std::vector<std::vector<Real> > _salientPeaksBins;
  std::vector<std::vector<Real> > _salientPeaksValues;
  std::vector<std::vector<Real> > _nonSalientPeaksBins;
  std::vector<std::vector<Real> > _nonSalientPeaksValues;

  Real _timeContinuityInFrames;
  Real _minDurationInFrames;
  Real _pitchContinuityInBins;
  size_t _numberFrames;
  Real _frameDuration;

  void removePeak(std::vector<std::vector<Real> >& peaksBins, std::vector<std::vector<Real> >& peaksValues, size_t i, int j);
  int findNextPeak(std::vector<std::vector<Real> >& peaksBins, std::vector<Real>& contoursBins, size_t i, bool backward=false);
  void trackPitchContour(size_t& index, std::vector<Real>& contourBins, std::vector <Real>& contourSaliences);

 public:
  PitchContours() {
    declareInput(_peakBins, "peakBins", "frame-wise array of cent bins corresponding to pitch salience function peaks");
    declareInput(_peakSaliences, "peakSaliences", "frame-wise array of values of salience function peaks");
    declareOutput(_contoursBins, "contoursBins", "array of frame-wise vectors of cent bin values representing each contour");
    declareOutput(_contoursSaliences, "contoursSaliences", "array of frame-wise vectors of pitch saliences representing each contour");
    declareOutput(_contoursStartTimes, "contoursStartTimes", "array of start times of each contour [s]");
    declareOutput(_duration, "duration", "time duration of the input signal [s]");
  }

  ~PitchContours() {
  };

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("hopSize", "the hop size with which the pitch salience function was computed", "(0,inf)", 128);
    declareParameter("binResolution", "salience function bin resolution [cents]", "(0,inf)", 10.0);
    declareParameter("peakFrameThreshold", "per-frame salience threshold factor (fraction of the highest peak salience in a frame)", "[0,1]", 0.9);
    declareParameter("peakDistributionThreshold", "allowed deviation below the peak salience mean over all frames (fraction of the standard deviation)", "[0,2]", 0.9);
    declareParameter("pitchContinuity", "pitch continuity cue (maximum allowed pitch change durig 1 ms time period) [cents]", "[0,inf)", 27.5625);
    declareParameter("timeContinuity", "time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]", "(0,inf)", 100.);
    declareParameter("minDuration", "the minimum allowed contour duration [ms]", "(0,inf)", 100.);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* version;
  static const char* description;

}; // class PitchContours

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PitchContours : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::vector<Real> > > _peakBins;
  Sink<std::vector<std::vector<Real> > > _peakSaliences;
  Source<std::vector<std::vector<Real> > > _contoursBins;
  Source<std::vector<std::vector<Real> > > _contoursSaliences;
  Source<std::vector<Real> > _contoursStartTimes;
  Source<Real> _duration;

 public:
  PitchContours() {
    declareAlgorithm("PitchContours");
    declareInput(_peakBins, TOKEN, "peakBins");
    declareInput(_peakSaliences, TOKEN, "peakSaliences");
    declareOutput(_contoursBins, TOKEN, "contoursBins");
    declareOutput(_contoursSaliences, TOKEN, "contoursSaliences");
    declareOutput(_contoursStartTimes, TOKEN, "contoursStartTimes");
    declareOutput(_duration, TOKEN, "duration");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PITCHCONTOURS_H
