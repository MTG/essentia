/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_BANDREJECT_H
#define ESSENTIA_BANDREJECT_H

#include "algorithmfactory.h"
#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace standard {


class BandReject : public Algorithm {

 protected:
  Input<std::vector<Real> > _x;
  Output<std::vector<Real> > _y;

  Algorithm* _filter;

 public:
  BandReject() {
    declareInput(_x, "signal", "the input signal");
    declareOutput(_y, "signal", "the filtered signal");

    _filter = AlgorithmFactory::create("IIR");
  }

  ~BandReject() {
    if(_filter) delete _filter;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("cutoffFrequency", "the cutoff frequency for the filter [Hz]", "(0,inf)", 1500.);
    declareParameter("bandwidth", "the bandwidth of the filter [Hz]", "(0,inf)", 500.);
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


class BandReject : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _x;
  Source<Real> _y;

  static const int preferredSize = 4096;

 public:
  BandReject() {
    declareAlgorithm("BandReject");
    declareInput(_x, STREAM, preferredSize, "signal");
    declareOutput(_y, STREAM, preferredSize, "signal");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BANDREJECT_H
