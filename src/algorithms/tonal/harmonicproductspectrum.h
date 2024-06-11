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

/*
 * This file is a port of the file harmonicproductspectrum.h from aubio,
 * http://aubio.piem.org/, in its version 0.3.2.
 *
 * The port was written by the author of aubio, Paul Brossier
 * <piem@altern.org>.
 */

#ifndef ESSENTIA_HARMONICPRODUCTSPECTRUM_H
#define ESSENTIA_HARMONICPRODUCTSPECTRUM_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class HarmonicProductSpectrum : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _pitch;
  Output<Real> _pitchConfidence;

  Algorithm* _peakDetect;

  Real _sampleRate;               /** sampling rate of the audio signal */
  int _frameSize;
  int _tauMin;
  int _tauMax;

 public:
  HarmonicProductSpectrum() {
    declareInput(_spectrum, "spectrum", "the input spectrum (preferably created with a hann window)");
    declareOutput(_pitch, "pitch", "detected pitch [Hz]");
    declareOutput(_pitchConfidence, "pitchConfidence", "confidence with which the pitch was detected [0,1]");

    _peakDetect = AlgorithmFactory::create("PeakDetection");
  }

  ~HarmonicProductSpectrum() {
    delete _peakDetect;
  };

  void declareParameters() {
    declareParameter("frameSize", "number of samples in the input spectrum", "[2,inf)", 2048);
    declareParameter("sampleRate", "sampling rate of the input spectrum [Hz]", "(0,inf)", 44100.);
    declareParameter("minFrequency", "the minimum allowed frequency [Hz]", "(0,inf)", 20.0);
    declareParameter("maxFrequency", "the maximum allowed frequency [Hz]", "(0,inf)", 22050.0);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

}; // class HarmonicProductSpectrum

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class HarmonicProductSpectrum : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _pitch;
  Source<Real> _pitchConfidence;

 public:
  HarmonicProductSpectrum() {
    declareAlgorithm("HarmonicProductSpectrum");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_pitch, TOKEN, "pitch");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_HARMONICPRODUCTSPECTRUM_H
