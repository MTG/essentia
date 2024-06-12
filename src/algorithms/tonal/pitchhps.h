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

#ifndef ESSENTIA_PITCHHPS_H
#define ESSENTIA_PITCHHPS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class PitchHPS : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _pitch;
  Output<Real> _pitchConfidence;

  Algorithm* _peakDetect;

  std::vector<Real> _positions;   /** peak positions */
  std::vector<Real> _amplitudes;  /** peak amplitudes */
  Real _sampleRate;               /** sampling rate of the audio signal */
  int _frameSize;
  int _tauMin;
  int _tauMax;
  int _numHarmonics;
  Real _magnitudeThreshold;

 public:
  PitchHPS() {
    declareInput(_spectrum, "spectrum", "the input spectrum (preferably created with a hann window)");
    declareOutput(_pitch, "pitch", "detected pitch [Hz]");
    declareOutput(_pitchConfidence, "pitchConfidence", "confidence with which the pitch was detected [0,1]");

    _peakDetect = AlgorithmFactory::create("PeakDetection");
  }

  ~PitchHPS() {
    delete _peakDetect;
  };

  void declareParameters() {
    declareParameter("frameSize", "number of samples in the input spectrum", "[2,inf)", 2048);
    declareParameter("sampleRate", "sampling rate of the input spectrum [Hz]", "(0,inf)", 44100.);
    declareParameter("minFrequency", "the minimum allowed frequency [Hz]", "(0,inf)", 20.0);
    declareParameter("maxFrequency", "the maximum allowed frequency [Hz]", "(0,inf)", 22050.0);
    declareParameter("numHarmonics", "number of harmonics to consider", "[1,inf)", 5);
    declareParameter("magnitudeThreshold", "threshold ratio for the amplitude filtering [0,1]", "[0,1]", 0.2);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

}; // class PitchHPS

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PitchHPS : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _pitch;
  Source<Real> _pitchConfidence;

 public:
  PitchHPS() {
    declareAlgorithm("PitchHPS");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_pitch, TOKEN, "pitch");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PITCHHPS_H
