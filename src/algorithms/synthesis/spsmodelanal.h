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

#ifndef ESSENTIA_SPSMODELANAL_H
#define ESSENTIA_SPSMODELANAL_H

#include "algorithm.h"
#include "algorithmfactory.h"
#include <fstream>


namespace essentia {
namespace standard {

class SpsModelAnal : public Algorithm {

 protected:
  Input<std::vector<std::complex<Real> > > _fft;
  Output<std::vector<Real> > _magnitudes;
  Output<std::vector<Real> > _frequencies;
  Output<std::vector<Real> > _phases;
  Output<std::vector<Real> > _stocenv;

  int _stocSize;
  Algorithm* _sineModelAnal;
  Algorithm* _sineModelSynth;

  // for resample
  Algorithm* _fftres;
  Algorithm* _ifftres;
  void initializeFFT(std::vector<std::complex<Real> >&fft, int sizeFFT);

// debug
std::ofstream _log;

 public:
  SpsModelAnal() {
    declareInput(_fft, "fft", "the input frame");
    declareOutput(_frequencies, "frequencies", "the frequencies of the sinusoidal peaks [Hz]");
    declareOutput(_magnitudes, "magnitudes", "the magnitudes of the sinusoidal peaks");
    declareOutput(_phases, "phases", "the phases of the sinusoidal peaks");
    declareOutput(_stocenv, "stocenv", "the stochastic envelope");

    _sineModelAnal = AlgorithmFactory::create("SineModelAnal");
    _sineModelSynth = AlgorithmFactory::create("SineModelSynth");

    // for resample
    _fftres = AlgorithmFactory::create("FFT");
    _ifftres = AlgorithmFactory::create("IFFT");

  }

  ~SpsModelAnal() {
//    delete _peakDetect;
//    delete _cartesianToPolar;
  delete _sineModelAnal;
  delete _sineModelSynth;

  // for resample
  delete _fftres;
  delete _ifftres;

  _log.close(); // debug
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("hopSize", "the hop size between frames", "[1,inf)", 512);
    declareParameter("fftSize", "the size of the internal FFT size (full spectrum size)", "[1,inf)", 2048);
    declareParameter("maxPeaks", "the maximum number of returned peaks", "[1,inf)", 100);
    declareParameter("maxFrequency", "the maximum frequency of the range to evaluate [Hz]", "(0,inf)", 5000.0);
    declareParameter("minFrequency", "the minimum frequency of the range to evaluate [Hz]", "[0,inf)", 0.0);
    declareParameter("magnitudeThreshold", "peaks below this given threshold are not outputted", "(-inf,inf)", 0.0);
    declareParameter("orderBy", "the ordering type of the outputted peaks (ascending by frequency or descending by magnitude)", "{frequency,magnitude}", "frequency");
    // sinusoidal tracking
    declareParameter("maxnSines", "maximum number of sines per frame", "(0,inf)", 100);
   // declareParameter("minSineDur", "minimum duration of sines in seconds", "(0,inf)", 0.01);
    declareParameter("freqDevOffset", "minimum frequency deviation at 0Hz", "(0,inf)", 20);
    declareParameter("freqDevSlope", "slope increase of minimum frequency deviation", "(-inf,inf)", 0.01);
    declareParameter("stocf", "decimation factor used for the stochastic approximation", "(0,1)", 0.2);

  }

  void configure();
  void compute();

  void stochasticModelAnal(const std::vector<std::complex<Real> > fftInput, const std::vector<Real> magnitudes, const std::vector<Real> frequencies, const std::vector<Real> phases, std::vector<Real> &stocEnv);
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

class SpsModelAnal : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::complex<Real> > > _fft; // input
  Source<std::vector<Real> > _frequencies;
  Source<std::vector<Real> > _magnitudes;
  Source<std::vector<Real> > _phases;
  Source<std::vector<Real> > _stocenv;

 public:
  SpsModelAnal() {
    declareAlgorithm("SpsModelAnal");
    declareInput(_fft, TOKEN, "fft");
    declareOutput(_frequencies, TOKEN, "frequencies");
    declareOutput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_phases, TOKEN, "phases");
    declareOutput(_stocenv, TOKEN, "stocenv");
  }
};

} // namespace streaming
} // namespace essentia




#endif // ESSENTIA_SPSMODELANAL_H
