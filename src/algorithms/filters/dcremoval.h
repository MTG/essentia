/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_DCREMOVAL_H
#define ESSENTIA_DCREMOVAL_H

#include "algorithmfactory.h"
#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace standard {

class DCRemoval : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _signalDC;

  Algorithm* _filter;

 public:
  DCRemoval() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_signalDC, "signal", "the filtered signal, with the DC component removed");

    _filter = AlgorithmFactory::create("IIR");
  }

  ~DCRemoval() {
    delete _filter;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("cutoffFrequency", "the cutoff frequency for the filter [Hz]", "(0,inf)", 40.);
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

class DCRemoval : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _signal;
  Source<Real> _signalDC;

  static const int preferredSize = 4096;

 public:
  DCRemoval() {
    declareAlgorithm("DCRemoval");
    declareInput(_signal, STREAM, preferredSize, "signal");
    declareOutput(_signalDC, STREAM, preferredSize, "signal");

    _signalDC.setBufferType(BufferUsage::forAudioStream);
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_DCREMOVAL_H
