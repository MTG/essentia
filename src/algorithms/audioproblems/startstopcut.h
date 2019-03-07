/*
 * Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef STARTSTOPCUT_H
#define STARTSTOPCUT_H

#include "algorithm.h"
#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class StartStopCut : public Algorithm {
 private:
  Input<std::vector<Real> > _audio;
  Output<int> _startCut;
  Output<int> _stopCut;

  Real _sampleRate;
  uint _hopSize, _frameSize;
  Real _maximumStartTime, _maximumStopTime;
  uint _maximumStartSamples, _maximumStopSamples;

  Real _threshold;

  Algorithm* _frameCutter;

 public:
  StartStopCut() {
    declareInput(_audio, "audio", "the input audio ");
    declareOutput(_startCut, "startCut", "1 if there is a cut at the begining of the audio");
    declareOutput(_stopCut, "stopCut", "1 if there is a cut at the end of the audio");
    
    _frameCutter = AlgorithmFactory::create("FrameCutter");
  }

  ~StartStopCut() {
    if (_frameCutter) delete _frameCutter;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sample rate", "(0,inf)", 44100.);
    declareParameter("frameSize", "the frame size for the internal power analysis", "(0,inf)", 256);
    declareParameter("hopSize", "the hop size for the internal power analysis", "(0,inf)", 256);
    declareParameter("threshold", "the threshold below which average energy is defined as silence [dB]", "(-inf,0])", -60);
    declareParameter("maximumStartTime", "if the first non-silent frame occurs before maximumStartTime startCut is activated [ms]", "[0,inf))", 10.0f);
    declareParameter("maximumStopTime", "if the last non-silent frame occurs after maximumStopTime to the end stopCut is activated [ms]", "[0,inf))", 10.0f);
  }

  void configure();
  void compute();
  void findNonSilentFrame(std::vector<Real> audio, int &nonSilentFrame, uint lastFrame);

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class StartStopCut : public StreamingAlgorithmWrapper {
 protected:
  Sink<std::vector<Real> > _audio;
  Source<int> _startCut;
  Source<int> _stopCut;

 public:
  StartStopCut() {
    declareAlgorithm("StartStopCut");
    declareInput(_audio, TOKEN, "audio");
    declareOutput(_startCut, TOKEN, "startCut");
    declareOutput(_stopCut, TOKEN, "stopCut");
  }
};

} // namespace streaming
} // namespace essentia


#endif // STARTSTOPCUT_H
