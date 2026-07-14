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

#ifndef ESSENTIA_BRIGHTNESS_H
#define ESSENTIA_BRIGHTNESS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class Brightness : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _brightness;


 public:
  Brightness() {
    // TODO: proper documentation
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_brightness, "brightness", "the brightness of the input signal");
  }

  ~Brightness() {
  }

  void declareParameters() {
    declareParameter("energyThreshold", "threshold below which to ignore the energy in a time window", "[0,inf)", 0.);
    declareParameter("ratioCrossover", "crossover frequency for calculating the HF energy ratio", "(0,inf)", 2000.);
    declareParameter("centroidCrossover", "highpass frequency for calculating the spectral centroid", "(0,inf)", 100.);
    declareParameter("hopSize", "step size for calculating spectrogram", "[0,inf)", 1024);
    declareParameter("windowSize", "block size (fft length) for calculating spectrogram", "[0,inf)", 2048);
    declareParameter("minFreq", "frequency for high-pass filtering audio prior to all analysis", "(0,inf)", 20.);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Brightness : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _signal;
  Source<Real> _brightness;

 public:
  Brightness() {
    declareAlgorithm("Brightness");
    declareInput(_signal, TOKEN, "signal");
    declareOutput(_brightness, TOKEN, "brightness");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BRIGHTNESS_H
