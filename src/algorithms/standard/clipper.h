/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef CLIPPER_H
#define CLIPPER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Clipper : public Algorithm {

 private:
  Input<std::vector<Real> > _input;
  Output<std::vector<Real> > _output;

  Real _max;
  Real _min;

 public:
  Clipper() : _max(0), _min(0) {
    declareInput(_input, "signal", "the input signal");
    declareOutput(_output, "signal", "the output signal with the added noise");
  }

  void declareParameters() {
    declareParameter("min", "the minimum value below which the signal will be clipped", "(-inf,inf)", -1.0);
    declareParameter("max", "the maximum value above which the signal will be clipped", "(-inf,inf)", 1.0);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace essentia
} // namespace standard

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Clipper : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _input;
  Source<Real> _output;

 public:
  Clipper() {
    // prefferred size should be kept as is, otherwise it may cause RhythmExtractor
    // to break. Raising this size too high may block rhythmextractor internal
    // network
    int preferredSize = 1;
    declareAlgorithm("Clipper");
    declareInput(_input, STREAM, preferredSize, "signal");
    declareOutput(_output, STREAM, preferredSize, "signal");

    _output.setBufferType(BufferUsage::forAudioStream);
  }
};

} // namespace streaming
} // namespace essentia


#endif // MAX_H
