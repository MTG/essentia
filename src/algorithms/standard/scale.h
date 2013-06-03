/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_SCALE_H
#define ESSENTIA_SCALE_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Scale : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _scaled;

  Real _factor, _maxValue;
  bool _clipping;

 public:
  Scale() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_scaled, "signal", "the output audio signal");
  }

  void declareParameters() {
    declareParameter("factor", "the multiplication factor by which the audio will be scaled", "[0,inf)", 10.0);
    declareParameter("clipping", "boolean flag whether to apply clipping or not", "{true,false}", true);
    declareParameter("maxAbsValue", "the maximum value above which to apply clipping", "[0,inf)", 1.0);
  }

  void configure();

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Scale : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _signal;
  Source<Real> _scaled;

 public:
  Scale() {
    int preferredSize = 4096;
    declareAlgorithm("Scale");
    declareInput(_signal, STREAM, preferredSize, "signal");
    declareOutput(_scaled, STREAM, preferredSize, "signal");

    _scaled.setBufferType(BufferUsage::forAudioStream);
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SCALE_H
