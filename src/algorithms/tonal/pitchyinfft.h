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

/*
 * This file is a port of the file pitchyinfft.h from aubio,
 * http://aubio.piem.org/, in its version 0.3.2.
 *
 * The port was written by the author of aubio, Paul Brossier
 * <piem@altern.org>.
 */

#ifndef ESSENTIA_PITCHYINFFT_H
#define ESSENTIA_PITCHYINFFT_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class PitchYinFFT : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _pitch;
  Output<Real> _pitchConfidence;

  Algorithm* _fft;
  Algorithm* _cart2polar;
  Algorithm* _peakDetect;

  std::vector<Real> _resPhase;    /** complex vector to compute square difference function */
  std::vector<Real> _resNorm;
  std::vector<Real> _sqrMag;      /** square difference function */
  std::vector<Real> _weight;      /** spectral weighting window (psychoacoustic model) */
  std::vector<Real> _yin;         /** Yin function */
  std::vector<Real> _positions;   /** autocorrelation peak positions */
  std::vector<Real> _amplitudes;  /** autocorrelation peak amplitudes */
  Real _sampleRate;               /** sampling rate of the audio signal */
  bool _interpolate;              /** whether to use peak interpolation */
  int _frameSize;
  //Real _tolerance;
  int _tauMin;
  int _tauMax;


 public:
  PitchYinFFT() {
    declareInput(_spectrum, "spectrum", "the input spectrum (preferably created with a hann window)");
    declareOutput(_pitch, "pitch", "detected pitch [Hz]");
    declareOutput(_pitchConfidence, "pitchConfidence", "confidence with which the pitch was detected [0,1]");

    _fft = AlgorithmFactory::create("FFT");
    _cart2polar = AlgorithmFactory::create("CartesianToPolar");
    _peakDetect = AlgorithmFactory::create("PeakDetection");
  }

  ~PitchYinFFT() {
    delete _fft;
    delete _cart2polar;
    delete _peakDetect;
  };

  void declareParameters() {
    declareParameter("frameSize", "number of samples in the input spectrum", "[2,inf)", 2048);
    declareParameter("sampleRate", "sampling rate of the input spectrum [Hz]", "(0,inf)", 44100.);
    declareParameter("minFrequency", "the minimum allowed frequency [Hz]", "(0,inf)", 20.0);
    declareParameter("maxFrequency", "the maximum allowed frequency [Hz]", "(0,inf)", 22050.0);
    declareParameter("interpolate", "boolean flag to enable interpolation", "{true,false}", true);
    //declareParameter("tolerance", "tolerance for peak detection", "[0,1]", 0.75);
  }

  void configure();
  void compute();

  void spectralWeights();

  static const char* name;
  static const char* description;

}; // class PitchYinFFT

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PitchYinFFT : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _pitch;
  Source<Real> _pitchConfidence;

 public:
  PitchYinFFT() {
    declareAlgorithm("PitchYinFFT");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_pitch, TOKEN, "pitch");
    declareOutput(_pitchConfidence, TOKEN, "pitchConfidence");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PITCHYINFFT_H
