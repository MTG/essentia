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

#ifndef ESSENTIA_PITCHCONTOURSMELODY_H
#define ESSENTIA_PITCHCONTOURSMELODY_H

#include <utility> // std::pair
#include "algorithmfactory.h"


namespace essentia {
namespace standard {

class PitchContoursMelody : public Algorithm {

 private:
  Input<std::vector<std::vector<Real> > > _contoursBins;
  Input<std::vector<std::vector<Real> > > _contoursSaliences;
  Input<std::vector<Real> > _contoursStartTimes;
  Input<Real> _duration;
  Output<std::vector<Real> > _pitch;
  Output<std::vector<Real> > _pitchConfidence;

  Algorithm* _movingAverage;
  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _spectrum;
  Algorithm* _spectralPeaks;

  Real _binResolution;
  Real _referenceFrequency;
  Real _voicingTolerance;
  bool _voiceVibrato;
  Real _sampleRate;
  int _hopSize;
  int _filterIterations;  // number of interations in the octave errors/pitch outliers filtering process
  bool _guessUnvoiced;

  Real _frameDuration;
  size_t _numberFrames;
  size_t _averagerShift;
  Real _outlierMaxDistance;
  Real _duplicateMaxDistance;
  Real _duplicateMinDistance;

  Real _vibratoPitchStddev;

  Real _minBin;
  Real _maxBin;

  // voice vibrato detection parameters
  //Real _vibratoSampleRate;
  int _vibratoFrameSize;
  int _vibratoHopSize;
  int _vibratoZeroPaddingFactor;
  int _vibratoFFTSize;
  Real _vibratoMinFrequency;
  Real _vibratoMaxFrequency;
  Real _vibratodBDropLobe;
  Real _vibratodBDropSecondPeak;

  Real _centToHertzBase;

  std::vector<size_t> _contoursStartIndices;
  std::vector<size_t> _contoursEndIndices;
  std::vector<Real> _contoursBinsMean;
  std::vector<Real> _contoursSaliencesTotal;  // total salience (sum of per-frame saliences)
  std::vector<Real> _contoursSaliencesMean;
  std::vector<Real> _contoursBinsStddev;

  std::vector<Real> _melodyPitchMean;               // melody pitch mean function
  std::vector <std::pair <int,int> > _duplicates;   // pairs of contour duplicates
  std::vector<size_t> _contoursSelected;    // indices of the selected contours
  std::vector<size_t> _contoursIgnored;     // indices of the ignored contours
  std::vector<size_t> _contoursSelectedInitially;
  std::vector<size_t> _contoursIgnoredInitially;
  size_t _numberContours;

  void voicingDetection(const std::vector<std::vector<Real> >& contoursBins,
                        const std::vector<std::vector<Real> >& contoursSaliences,
                        const std::vector<Real>& contoursStartTimes);
  bool detectVoiceVibrato(std::vector<Real> contourBins, const Real binMean);
  void computeMelodyPitchMean(const std::vector<std::vector<Real> >& contoursBins);
  void detectContourDuplicates(const std::vector<std::vector<Real> >& contoursBins);
  void removeContourDuplicates();
  void removePitchOutliers();

 public:
  PitchContoursMelody() {
    declareInput(_contoursBins, "contoursBins", "array of frame-wise vectors of cent bin values representing each contour");
    declareInput(_contoursSaliences, "contoursSaliences", "array of frame-wise vectors of pitch saliences representing each contour");
    declareInput(_contoursStartTimes, "contoursStartTimes", "array of the start times of each contour [s]");
    declareInput(_duration, "duration", "time duration of the input signal [s]");
    declareOutput(_pitch, "pitch", "vector of estimated pitch values (i.e., melody) [Hz]");
    declareOutput(_pitchConfidence, "pitchConfidence", "confidence with which the pitch was detected");

    _movingAverage = AlgorithmFactory::create("MovingAverage");
    _frameCutter = AlgorithmFactory::create("FrameCutter");
    _windowing = AlgorithmFactory::create("Windowing");
    _spectrum = AlgorithmFactory::create("Spectrum");
    _spectralPeaks = AlgorithmFactory::create("SpectralPeaks");
  }

  ~PitchContoursMelody() {
    delete _movingAverage;
    delete _frameCutter;
    delete _windowing;
    delete _spectrum;
    delete _spectralPeaks;
  };

  void declareParameters() {
    declareParameter("referenceFrequency", "the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin", "(0,inf)", 55.0);
    declareParameter("binResolution", "salience function bin resolution [cents]", "(0,inf)", 10.0);
    declareParameter("sampleRate", "the sampling rate of the audio signal (Hz)", "(0,inf)", 44100.);
    declareParameter("hopSize", "the hop size with which the pitch salience function was computed", "(0,inf)", 128);
    declareParameter("voicingTolerance", "allowed deviation below the average contour mean salience of all contours (fraction of the standard deviation)", "[-1.0,1.4]", 0.2);
    declareParameter("voiceVibrato", "detect voice vibrato", "{true,false}", false);
    declareParameter("filterIterations", "number of interations for the octave errors / pitch outlier filtering process", "[1,inf)", 3);
    declareParameter("guessUnvoiced", "Estimate pitch for non-voiced segments by using non-salient contours when no salient ones are present in a frame", "{false,true}", false);
    declareParameter("minFrequency", "the minimum allowed frequency for salience function peaks (ignore contours with peaks below) [Hz]", "[0,inf)", 80.0);
    declareParameter("maxFrequency", "the minimum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz]", "[0,inf)", 20000.0); // just some large value greater than anything we would need
  }

  void configure();
  void compute();

  static const char* name;
  static const char* version;
  static const char* description;

}; // class PitchContoursMelody

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PitchContoursMelody : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::vector<Real> > > _contoursBins;
  Sink<std::vector<std::vector<Real> > > _contoursSaliences;
  Sink<std::vector<Real> > _contoursStartTimes;
  Sink<Real> _duration;
  Source<std::vector<Real> > _pitch;
  Source<std::vector<Real> > _pitchConfidence;

 public:
  PitchContoursMelody() {
    declareAlgorithm("PitchContoursMelody");
    declareInput(_contoursBins, TOKEN, "contoursBins");
    declareInput(_contoursSaliences, TOKEN, "contoursSaliences");
    declareInput(_contoursStartTimes, TOKEN, "contoursStartTimes");
    declareInput(_duration, TOKEN, "duration");
    declareOutput(_pitch, TOKEN, "pitch");
    declareOutput(_pitchConfidence, TOKEN, "pitchConfidence");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PITCHCONTOURSMELODY_H
