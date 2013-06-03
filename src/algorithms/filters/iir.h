/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_IIR_H
#define ESSENTIA_IIR_H

#include "algorithmfactory.h"
#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace standard {

class IIR : public Algorithm {

 private:
  Input<std::vector<Real> > _x;
  Output<std::vector<Real> > _y;

  std::vector<Real> _a;
  std::vector<Real> _b;
  std::vector<Real> _state;

 public:
  IIR() {
    declareInput(_x, "signal", "the input signal");
    declareOutput(_y, "signal", "the filtered signal");
  }

  void declareParameters() {
    std::vector<Real> defaultParam(1, 1.0);
    declareParameter("numerator", "the list of coefficients of the numerator. Often referred to as the B coefficient vector.", "", defaultParam);
    declareParameter("denominator", "the list of coefficients of the denominator. Often referred to as the A coefficient vector.", "", defaultParam);
  }


  void configure();
  void compute();

  void reset();

  static const char* name;
  static const char* description;

};

} // namespace standard
namespace streaming {

class IIR : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _x;
  Source<Real> _y;

  static const int preferredSize = 4096;

 public:
  IIR() {
    declareAlgorithm("IIR");
    declareInput(_x, STREAM, preferredSize, "signal");
    declareOutput(_y, STREAM, preferredSize, "signal");

    _y.setBufferType(BufferUsage::forAudioStream);
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_IIR_H
