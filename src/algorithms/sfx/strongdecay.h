/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_STRONGDECAY_H
#define ESSENTIA_STRONGDECAY_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class StrongDecay : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _strongDecay;
  Algorithm* _centroid;
  Algorithm* _abs;

 public:
  StrongDecay() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_strongDecay, "strongDecay", "the strong decay");

    _centroid = AlgorithmFactory::create("Centroid");
    _abs = AlgorithmFactory::create("UnaryOperator", "type", "abs");
  }

  ~StrongDecay() {
    delete _centroid;
    delete _abs;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void compute();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia


#include "accumulatoralgorithm.h"

namespace essentia {
namespace streaming {

class StrongDecay : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _signal;
  Source<Real> _strongDecay;

  double _centroid;
  double _energy;
  double _weights;
  int _idx;

 public:
  StrongDecay() {
    declareInputStream(_signal, "signal", "the input audio signal");
    declareOutputResult(_strongDecay, "strongDecay", "the strong decay");
    reset();
  }

  void reset();
  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void consume();
  void finalProduce();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_STRONGDECAY_H
