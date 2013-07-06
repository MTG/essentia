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

#ifndef ESSENTIA_PREDOMINANTMELODY_H
#define ESSENTIA_PREDOMINANTMELODY_H

#include "algorithmfactory.h"
#include "network.h"

namespace essentia {
namespace standard {

class PredominantMelody : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _pitch;
  Output<std::vector<Real> > _pitchConfidence;

  // Pre-processing
  Algorithm* _frameCutter;
  Algorithm* _windowing;

  // Spectral peaks
  Algorithm* _spectrum;
  Algorithm* _spectralPeaks;

  // Pitch salience contours
  Algorithm* _pitchSalienceFunction;
  Algorithm* _pitchSalienceFunctionPeaks;
  Algorithm* _pitchContours;

  // Melody
  Algorithm* _pitchContoursMelody;

 public:
  PredominantMelody() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_pitch, "pitch", "the estimated pitch values [Hz]");
    declareOutput(_pitchConfidence, "pitchConfidence", "confidence with which the pitch was detected");

    // Pre-processing
    _frameCutter = AlgorithmFactory::create("FrameCutter");
    _windowing = AlgorithmFactory::create("Windowing");

    // Spectral peaks
    _spectrum = AlgorithmFactory::create("Spectrum");
    _spectralPeaks = AlgorithmFactory::create("SpectralPeaks");

    // Pitch salience contours
    _pitchSalienceFunction = AlgorithmFactory::create("PitchSalienceFunction");
    _pitchSalienceFunctionPeaks = AlgorithmFactory::create("PitchSalienceFunctionPeaks");
    _pitchContours = AlgorithmFactory::create("PitchContours");

    // Melody
    _pitchContoursMelody = AlgorithmFactory::create("PitchContoursMelody");
  }

  ~PredominantMelody();

  void declareParameters() {
    // pre-processing
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size for computing pitch saliecnce", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size with which the pitch salience function was computed", "(0,inf)", 128);

    // pitch salience function
    declareParameter("binResolution", "salience function bin resolution [cents]", "(0,inf)", 10.0);
    declareParameter("referenceFrequency", "the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin", "(0,inf)", 55.0);
    declareParameter("magnitudeThreshold", "spectral peak magnitude threshold (maximum allowed difference from the highest peak in dBs)", "[0,inf)",  40);
    declareParameter("magnitudeCompression", "magnitude compression parameter for the salience function (=0 for maximum compression, =1 for no compression)", "(0,1]", 1.0);
    declareParameter("numberHarmonics", "number of considered harmonics", "[1,inf)", 20);
    declareParameter("harmonicWeight", "harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay)", "(0,1)", 0.8);

    // pitch contour tracking
    declareParameter("peakFrameThreshold", "per-frame salience threshold factor (fraction of the highest peak salience in a frame)", "[0,1]", 0.9);
    declareParameter("peakDistributionThreshold", "allowed deviation below the peak salience mean over all frames (fraction of the standard deviation)", "[0,2]", 0.9);
    declareParameter("pitchContinuity", "pitch continuity cue (maximum allowed pitch change during 1 ms time period) [cents]", "[0,inf)", 27.5625);
    declareParameter("timeContinuity", "time continuity cue (the maximum allowed gap duration for a pitch contour) [ms]", "(0,inf)", 100);
    declareParameter("minDuration", "the minimum allowed contour duration [ms]", "(0,inf)", 100);

    // melody detection
    declareParameter("voicingTolerance", "allowed deviation below the average contour mean salience of all contours (fraction of the standard deviation)", "[-1.0,1.4]", 0.2);
    declareParameter("voiceVibrato", "detect voice vibrato", "{true,false}", false);
    declareParameter("filterIterations", "number of iterations for the octave errors / pitch outlier filtering process", "[1,inf)", 3);
    declareParameter("guessUnvoiced", "estimate pitch for non-voiced segments by using non-salient contours when no salient ones are present in a frame", "{false,true}", false);
    declareParameter("minFrequency", "the minimum allowed frequency for salience function peaks (ignore contours with peaks below) [Hz]", "[0,inf)", 80.0);
    declareParameter("maxFrequency", "the minimum allowed frequency for salience function peaks (ignore contours with peaks above) [Hz]", "[0,inf)", 20000.0); // just some large value greater than anything we would need
  }


  void compute();
  void configure();

  void reset() {
    _frameCutter->reset();
  }

  static const char* name;
  static const char* version;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#include "pool.h"
#include "streamingalgorithmcomposite.h"

namespace essentia {
namespace streaming {

class PredominantMelody : public AlgorithmComposite {

 protected:
  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _spectrum;
  Algorithm* _spectralPeaks;
  Algorithm* _pitchSalienceFunction;
  Algorithm* _pitchSalienceFunctionPeaks;
  standard::Algorithm* _pitchContours;
  standard::Algorithm* _pitchContoursMelody;

  SinkProxy<Real> _signal;
  Source<std::vector<Real> > _pitch;
  Source<std::vector<Real> > _pitchConfidence;

  Pool _pool;
  Algorithm* _poolStorageBins;
  Algorithm* _poolStorageValues;

  scheduler::Network* _network;

 public:
  PredominantMelody();
   ~PredominantMelody();

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
    declareProcessStep(SingleShot(this));
  }

  void declareParameters() {
    // pre-processing
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size for computing pitch saliecnce", "(0,inf)", 2048);
    declareParameter("hopSize", "the hop size with which the pitch salience function was computed", "(0,inf)", 128);

    // pitch salience function
    declareParameter("binResolution", "salience function bin resolution [cents]", "(0,inf)", 10.0);
    declareParameter("referenceFrequency", "the reference frequency for Hertz to cent convertion [Hz], corresponding to the 0th cent bin", "(0,inf)", 55.0);
    declareParameter("magnitudeThreshold", "peak magnitude threshold (maximum allowed difference from the highest peak in dBs)", "[0,inf)",  40);
    declareParameter("magnitudeCompression", "magnitude compression parameter (=0 for maximum compression, =1 for no compression)", "(0,1]", 1.0);
    declareParameter("numberHarmonics", "number of considered hamonics", "[1,inf)", 20);
    declareParameter("harmonicWeight", "harmonic weighting parameter (weight decay ratio between two consequent harmonics, =1 for no decay)", "(0,1)", 0.8);

    // pitch salience function peaks
    declareParameter("minFrequency", "the minimum allowed frequency for salience function peaks (ignore peaks below) [Hz]", "[0,inf)", 80.0);
    declareParameter("maxFrequency", "the maximum allowed frequency for salience function peaks (ignore peaks above) [Hz]", "[0,inf)", 20000.0); // just some very large value

    // pitch contour tracking
    declareParameter("peakFrameThreshold", "per-frame salience threshold factor (fraction of the highest peak salience in a frame)", "[0,1]", 0.9);
    declareParameter("peakDistributionThreshold", "allowed deviation below the peak salience mean over all frames (fraction of the standard deviation)", "[0,1]", 0.9);
    declareParameter("pitchContinuity", "pitch continuity cue (maximum allowed pitch change durig 1 ms time period) [cents]", "[0,inf)", 27.5625);
    declareParameter("timeContinuity", "tine continuity cue (the maximum allowed gap duration for a pitch contour) [ms]", "(0,inf)", 100);
    declareParameter("minDuration", "the minimum allowed contour duration [ms]", "(0,inf)", 100);

    // melody detection
    declareParameter("voicingTolerance", "allowed deviation below the average contour mean salience of all contours (fraction of the standard deviation)", "[-1.0,1.4]", 0.2);
    declareParameter("voiceVibrato", "detect voice vibrato", "{true,false}", false);
    declareParameter("filterIterations", "number of interations for the octave errors / pitch outlier filtering process", "[1,inf)", 3);
    declareParameter("guessUnvoiced", "guess pitch using non-salient contours when no salient ones are present in a frame", "{false,true}", false);
  };

  void configure();
  AlgorithmStatus process();
  void reset();
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PREDOMINANTMELODY_H
