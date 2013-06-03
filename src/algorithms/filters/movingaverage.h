/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_MOVINGAVERAGE_H
#define ESSENTIA_MOVINGAVERAGE_H

#include "algorithmfactory.h"
#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace standard {

class MovingAverage : public Algorithm {

 protected:
  Input<std::vector<Real> > _x;
  Output<std::vector<Real> > _y;

  Algorithm* _filter;

 public:
  MovingAverage() {
    declareInput(_x, "signal", "the input audio signal");
    declareOutput(_y, "signal", "the filtered signal");

    _filter = AlgorithmFactory::create("IIR");
  }

  ~MovingAverage() {
    if (_filter) delete _filter;
  }

  void declareParameters() {
    declareParameter("size", "the size of the window [audio samples]", "(1,inf)", 6);
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

class MovingAverage : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _x;
  Source<Real> _y;

  static const int preferredSize = 4096;

 public:
  MovingAverage() {
    declareAlgorithm("MovingAverage");
    declareInput(_x, STREAM, preferredSize, "signal");
    declareOutput(_y, STREAM, preferredSize, "signal");

    _y.setBufferType(BufferUsage::forAudioStream);
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_MOVINGAVERAGE_H
