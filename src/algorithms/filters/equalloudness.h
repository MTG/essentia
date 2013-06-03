/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_EQUALLOUDNESS_H
#define ESSENTIA_EQUALLOUDNESS_H

#include "algorithmfactory.h"
#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace standard {

class EqualLoudness : public Algorithm {

 protected:
  Input<std::vector<Real> > _x;
  Output<std::vector<Real> > _y;

  std::vector<Real> _z; // intermediate storage vector

  Algorithm* _yulewalkFilter;
  Algorithm* _butterworthFilter;

 public:
  EqualLoudness() {
    declareInput(_x, "signal", "the input signal");
    declareOutput(_y, "signal", "the filtered signal");

    _yulewalkFilter = AlgorithmFactory::create("IIR");
    _butterworthFilter = AlgorithmFactory::create("IIR");
  }

  ~EqualLoudness() {
    delete _yulewalkFilter;
    delete _butterworthFilter;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "{32000,44100,48000}", 44100.);
  }

  void reset();
  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
namespace streaming {

class EqualLoudness : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _x;
  Source<Real> _y;

  static const int preferredSize = 4096;

 public:
  EqualLoudness() {
    declareAlgorithm("EqualLoudness");
    declareInput(_x, STREAM, preferredSize, "signal");
    declareOutput(_y, STREAM, preferredSize, "signal");

    _y.setBufferType(BufferUsage::forLargeAudioStream);
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_EQUALLOUDNESS_H
