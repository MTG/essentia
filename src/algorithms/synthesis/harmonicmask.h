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

#ifndef ESSENTIA_HARMONICMASK_H
#define ESSENTIA_HARMONICMASK_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class HarmonicMask : public Algorithm {

 private:
    Input<std::vector<std::complex<Real> > > _fft;
    Input<Real> _pitch;
    Output<std::vector<std::complex<Real> > > _outfft;


  int _sampleRate;
  int _binWidth;
  Real _attenuation;
  Real _gainLin;

 public:
  HarmonicMask() {

    declareInput(_fft, "fft", "the input frame");
    declareInput(_pitch, "pitch", "an estimate of the fundamental frequency of the signal [Hz]");
    declareOutput(_outfft, "fft", "the output frame");


  }

  ~HarmonicMask() {

  }

  void declareParameters() {
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("binWidth", "number of bins per harmonic partials applied to the mask. This will depend on the internal FFT size", "[0,inf)", 4);
    declareParameter("attenuation", "attenuation in dB's of the muted pitched component. If value is positive the pitched component is attenuated (muted), if the value is negative the pitched component is soloed (i.e. background component is attenuated).", "[-inf,inf)", -200.);
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

class HarmonicMask : public StreamingAlgorithmWrapper {

 protected:

  int _sampleRate;
  int _binWidth;
  Real _attenuation;
  Real _attenuationLin;


  Sink<std::vector<std::complex<Real> > > _fft; // input
  Sink<Real> _pitch; // input  for YinFFT
  Source<std::vector<std::complex<Real> > > _outfft;


 public:
  HarmonicMask() {
    declareAlgorithm("HarmonicMask");
    declareInput(_fft, TOKEN, "fft");
    declareInput(_pitch, TOKEN, "pitch");
    declareOutput(_outfft, TOKEN, "fft");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_HARMONICMASK_H
