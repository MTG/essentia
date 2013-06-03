/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_METER_H
#define ESSENTIA_METER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Meter : public Algorithm {

 private:
  Input<std::vector<std::vector<Real> > > _beatogram;
  Output<Real> _meter;

 public:
  Meter() {
    declareInput(_beatogram, "beatogram", "filtered matrix loudness");
    declareOutput(_meter, "meter", "the time signature");
  }

  ~Meter() {}

  void declareParameters() {}

  void compute();
  void configure();

  static const char* name;
  static const char* version;
  static const char* description;

 private:
  bool isPowerN(int val, int power);
  bool isPowerHarmonic(int x, int y);
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Meter : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::vector<Real> > > _beatogram;
  Source<Real> _meter;

 public:
  Meter() {
    declareAlgorithm("Meter");
    declareInput(_beatogram, TOKEN, "beatogram");
    declareOutput(_meter, TOKEN, "meter");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_METER_H
