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

#ifndef ESSENTIA_STOCHASTICMODELANAL_H
#define ESSENTIA_STOCHASTICMODELANAL_H

#include "algorithm.h"
#include "algorithmfactory.h"
#include <fstream>


namespace essentia {
namespace standard {

class StochasticModelAnal : public Algorithm {

 protected:
  //Input<std::vector<std::complex<Real> > > _fft;
  Input<std::vector<Real> > _frame;
  Output<std::vector<Real> > _magnitudes;
  Output<std::vector<Real> > _frequencies;
  Output<std::vector<Real> > _phases;
  Output<std::vector<Real> > _stocenv;

  int _stocSize;
  Algorithm* _window;
  Algorithm* _fft;
  Algorithm* _sineModelAnal;
  Algorithm* _sineSubtraction;
  Algorithm* _sineModelSynth;

  std::vector<Real> _stocFrameIn; // input frame for the stochaastic analysis algorithm

  // for resample
//  Algorithm* _fftres;
//  Algorithm* _ifftres;
  void initializeFFT(std::vector<std::complex<Real> >&fft, int sizeFFT);

// debug
std::ofstream _log;

 public:
  StochasticModelAnal() {
    declareInput(_frame, "frame", "the input frame");
    //declareInput(_fft, "fft", "the input frame");
    declareOutput(_frequencies, "frequencies", "the frequencies of the sinusoidal peaks [Hz]");
    declareOutput(_magnitudes, "magnitudes", "the magnitudes of the sinusoidal peaks");
    declareOutput(_phases, "phases", "the phases of the sinusoidal peaks");
    declareOutput(_stocenv, "stocenv", "the stochastic envelope");


    _window = AlgorithmFactory::create("Windowing");
    _fft = AlgorithmFactory::create("FFT");
    _sineModelAnal = AlgorithmFactory::create("SineModelAnal");
    _sineSubtraction = AlgorithmFactory::create("SineSubtraction");

    _stochasticModelAnal = AlgorithmFactory::create("StochasticModelAnal");

    _sineModelSynth = AlgorithmFactory::create("SineModelSynth");

    // for resample
//    _fftres = AlgorithmFactory::create("FFT");
//    _ifftres = AlgorithmFactory::create("IFFT");

  }

  ~StochasticModelAnal() {
//    delete _peakDetect;
//    delete _cartesianToPolar;
  delete _window;
  delete _fft;
  delete _sineModelAnal;
  delete _sineSubtraction;
  delete _sineModelSynth;


  // for resample
//  delete _fftres;
//  delete _ifftres;

  _log.close(); // debug
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("hopSize", "the hop size between frames", "[1,inf)", 512);
    declareParameter("fftSize", "the size of the internal FFT size (full spectrum size)", "[1,inf)", 2048);
    declareParameter("stocf", "decimation factor used for the stochastic approximation", "(0,1]", 0.2);

  }

  void configure();
  void compute();

 // void stochasticModelAnal(const std::vector<std::complex<Real> > fftInput, const std::vector<Real> magnitudes, const std::vector<Real> frequencies, const std::vector<Real> phases, std::vector<Real> &stocEnv);
  void resample(const std::vector<Real> in, std::vector<Real> &out, const int sizeOut);

  static const char* name;
  static const char* description;



 private:

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class StochasticModelAnal : public StreamingAlgorithmWrapper {

 protected:
  //Sink<std::vector<std::complex<Real> > > _fft; // input
  Sink<std::vector<Real> > _frame; // input
  Source<std::vector<Real> > _stocenv;

 public:
  StochasticModelAnal() {
    declareAlgorithm("StochasticModelAnal");
    declareInput(_frame, TOKEN, "frame");
    //declareInput(_fft, TOKEN, "fft");
    declareOutput(_stocenv, TOKEN, "stocenv");
  }
};

} // namespace streaming
} // namespace essentia




#endif // ESSENTIA_STOCHASTICMODELANAL_H
