/*
 * Copyright (C) 2006-2015  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_SINESUBTRACTION_H
#define ESSENTIA_SINESUBTRACTION_H

// defines for generateSine function
#define BH_SIZE 1001
#define BH_SIZE_BY2 501
#define MFACTOR 100


#include "algorithm.h"
#include "algorithmfactory.h"


namespace essentia {
namespace standard {

class SineSubtraction : public Algorithm {

 private:
  Input<std::vector<Real> > _inframe;
  Input<std::vector<Real> > _magnitudes;
  Input<std::vector<Real> > _frequencies;
  Input<std::vector<Real> > _phases;
  Output<std::vector<Real> > _outframe;

  Real _sampleRate;
  int _fftSize;
  int _hopSize;

  std::vector<Real> _synwindow;

  std::vector<Real> _lastytfreq;
  std::vector<Real> _lastytphase;

  Algorithm* _window;
  Algorithm* _fft;
  Algorithm* _ifft;
  Algorithm* _overlapadd;

  void initializeFFT(std::vector<std::complex<Real> >&fft, int sizeFFT);
  void subtractFFT(std::vector<std::complex<Real> >&fft1, const std::vector<std::complex<Real> >&fft2);
  void generateSines(const std::vector<Real> magnitudes, const std::vector<Real> frequencies, const std::vector<Real> phases, std::vector<std::complex<Real> >&outfft);
  void createSynthesisWindow(std::vector<Real> &synwindow, int hopSize, int winSize);
  void applySynthesisWindow(std::vector<Real> &inframe, const std::vector<Real> synwindow);

 public:
  SineSubtraction() {
    declareInput(_inframe, "frame", "the input audio frame to subtract from");
    declareInput(_magnitudes, "magnitudes", "the magnitudes of the sinusoidal peaks");
    declareInput(_frequencies, "frequencies", "the frequencies of the sinusoidal peaks [Hz]");
    declareInput(_phases, "phases", "the phases of the sinusoidal peaks");
    declareOutput(_outframe, "frame", "the output audio frame");

    _window = AlgorithmFactory::create("Windowing");
    _fft = AlgorithmFactory::create("FFT");
    _ifft = AlgorithmFactory::create("IFFT");
    _overlapadd = AlgorithmFactory::create("OverlapAdd");

  }

    ~SineSubtraction() {
    delete _window;
    delete _fft;
    delete _ifft;
    delete _overlapadd;
  }

  void declareParameters() {
    declareParameter("fftSize", "the size of the FFT internal process (full spectrum size) and output frame. Minimum twice the hopsize.", "[1,inf)", 512);
    declareParameter("hopSize", "the hop size between frames", "[1,inf)", 128);
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
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

class SineSubtraction : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _inframe;
  Sink<std::vector<Real> > _magnitudes;
  Sink<std::vector<Real> > _frequencies;
  Sink<std::vector<Real> > _phases;
  Source<std::vector<Real> > _outframe;


 public:
  SineSubtraction() {
    declareAlgorithm("SineSubtraction");
    declareInput(_inframe, TOKEN, "frame");
    declareInput(_magnitudes, TOKEN, "magnitudes");
    declareInput(_frequencies, TOKEN, "frequencies");
    declareInput(_phases, TOKEN, "phases");
    declareOutput(_outframe, TOKEN, "frame");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_SINESUBTRACTION_H
