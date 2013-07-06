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

#ifndef ESSENTIA_PANNING_H
#define ESSENTIA_PANNING_H

#include "algorithmfactory.h"
#include "tnt/tnt.h"
#include <complex>

namespace essentia {
namespace standard {

class Panning : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrumLeft;
  Input<std::vector<Real> >_spectrumRight;
  Output<TNT::Array2D<Real> > _panningCoeffs;

  int _averageFrames;
  int _panningBins;
  int _numCoeffs;
  int _numBands;
  Real _sampleRate;
  bool _warpedPanorama;
  std::vector<Real> _histogramAccumulated;
  int _nFrames;

  Algorithm* _ifft;

 public:
  Panning() {
    declareInput(_spectrumLeft, "spectrumLeft", "Left channel's spectrum");
    declareInput(_spectrumRight, "spectrumRight", "Right channel's spectrum");
    declareOutput(_panningCoeffs, "panningCoeffs", "Parameters that define the panning curve at each frame");

    // Pre-processing
    _ifft = AlgorithmFactory::create("IFFT");
  }

  ~Panning(){
    delete _ifft;
  }

  void declareParameters() {
    declareParameter("averageFrames", "number of frames to take into account for averaging", "[0,inf)", 43);
    declareParameter("panningBins", "size of the histogram of ratios (in bins)", "(1,inf)", 512);
    declareParameter("numCoeffs", "number of coefficients used to define the panning curve at each frame", "(0,inf)", 20);
    declareParameter("numBands", "number of mel bands", "[1,inf)", 1);
    declareParameter("warpedPanorama", "if true, warped panorama is applied, having more resolution in the center area", "{false,true}", true);
    declareParameter("sampleRate", "audio sampling rate [Hz]", "(0,inf)", 44100.);
  }

  void compute();
  void configure();
  void reset();

  static const char* name;
  static const char* description;

 protected:

  void calculateHistogram(const std::vector<Real>& specL, const std::vector<Real>& specR, std::vector<Real>& ratios, std::vector<Real>& result );
  void calculateCoefficients(const std::vector<Real>& histAcum, std::vector<std::complex<Real> >& coeffs );
  void correctAudibleAngle(std::vector<Real>& ratios);
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Panning : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrumLeft;
  Sink<std::vector<Real> > _spectrumRight;
  Source<TNT::Array2D<Real> > _panningCoeffs;

 public:
  Panning() {
    declareAlgorithm("Panning");
    declareInput(_spectrumLeft, TOKEN, "spectrumLeft");
    declareInput(_spectrumRight, TOKEN, "spectrumRight");
    declareOutput(_panningCoeffs, TOKEN, "panningCoeffs");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PANNING_H
