/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
