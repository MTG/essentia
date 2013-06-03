/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_ALLPASS_H
#define ESSENTIA_ALLPASS_H

#include "algorithmfactory.h"
#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace standard {


class AllPass : public Algorithm {

 protected:
  Input<std::vector<Real> > _x;
  Output<std::vector<Real> > _y;

  Algorithm* _filter;

 public:
  AllPass() {
    declareInput(_x, "signal", "the input signal");
    declareOutput(_y, "signal", "the filtered signal");

    _filter = AlgorithmFactory::create("IIR");
  }

  ~AllPass() {
    if (_filter) delete _filter;
  }

  void declareParameters() {
    declareParameter("order", "the order of the filter", "{1,2}", 1);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("cutoffFrequency", "the cutoff frequency for the filter [Hz]", "(0,inf)", 1500.);
    declareParameter("bandwidth", "the bandwidth of the filter [Hz] (used only for 2nd-order filters)", "(0,inf)", 500.);
  }

  void reset() {
    _filter->reset();
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};


} // namespace standard
namespace streaming {


class AllPass : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _x;
  Source<Real> _y;

  static const int preferredSize = 4096;

 public:
  AllPass() {
    declareAlgorithm("AllPass");
    declareInput(_x, STREAM, preferredSize, "signal");
    declareOutput(_y, STREAM, preferredSize, "signal");

    _y.setBufferType(BufferUsage::forAudioStream);
  }
};


} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_ALLPASS_H
