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

#ifndef ESSENTIA_PITCHYIN_H
#define ESSENTIA_PITCHYIN_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class PitchYin : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _pitch;
  Output<Real> _pitchConfidence;

  Algorithm* _peakDetectLocal;
  Algorithm* _peakDetectGlobal;

  std::vector<Real> _yin;         // Yin function (cumulative mean normalized difference)
  std::vector<Real> _positions;   // Yin function peak positions
  std::vector<Real> _amplitudes;  // Yin function peak amplitudes

  int _frameSize;
  Real _sampleRate;               
  bool _interpolate;  // whether to use peak interpolation
  Real _tolerance;
  int _tauMin;
  int _tauMax;


 public:
  PitchYin() {
    declareInput(_signal, "signal", "the input signal frame");
    declareOutput(_pitch, "pitch", "detected pitch [Hz]");
    declareOutput(_pitchConfidence, "pitchConfidence", "confidence with which the pitch was detected [0,1]");

    _peakDetectLocal = AlgorithmFactory::create("PeakDetection");
    _peakDetectGlobal = AlgorithmFactory::create("PeakDetection");
  }

  ~PitchYin() {
    delete _peakDetectLocal;
    delete _peakDetectGlobal;
  };

  void declareParameters() {
    declareParameter("frameSize", "number of samples in the input frame (this is an optional parameter to optimize memory allocation)", "[2,inf)", 2048);
    declareParameter("sampleRate", "sampling rate of the input audio [Hz]", "(0,inf)", 44100.);
    declareParameter("minFrequency", "the minimum allowed frequency [Hz]", "(0,inf)", 20.0);
    declareParameter("maxFrequency", "the maximum allowed frequency [Hz]", "(0,inf)", 22050.0);
    declareParameter("interpolate", "enable interpolation", "{true,false}", true);
    declareParameter("tolerance", "tolerance for peak detection", "[0,1]", 0.15);
    // NOTE: default tolerance value is taken from aubio yin implementation
    // https://github.com/piem/aubio/blob/master/src/pitch/pitchyin.c
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

}; // class PitchYin

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PitchYin : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _pitch;
  Source<Real> _pitchConfidence;

 public:
  PitchYin() {
    declareAlgorithm("PitchYin");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_pitch, TOKEN, "pitch");
    declareOutput(_pitchConfidence, TOKEN, "pitchConfidence");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PITCHYIN_H
