/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_BEATOGRAM_H
#define ESSENTIA_BEATOGRAM_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Beatogram : public Algorithm {

 private:
  Input<std::vector<Real> > _loudness;
  Input<std::vector<std::vector<Real> > > _loudnessBandRatio;
  Output<std::vector<std::vector<Real> > > _beatogram;

  int _windowSize;

 public:
  Beatogram() {
    declareInput(_loudness, "loudness", "the loudness at each beat");
    declareInput(_loudnessBandRatio, "loudnessBandRatio", "matrix of loudness ratios at each band and beat");
    declareOutput(_beatogram, "beatogram", "filtered matrix loudness");
  }

  ~Beatogram() {}

  void declareParameters() {
    declareParameter("size", "number of beats for dynamic filtering", "[1,inf)", 16);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* version;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Beatogram : public StreamingAlgorithmWrapper {

  // TODO: it should be implemented apropriately for a pure streaming algo

 protected:
  Sink<std::vector<Real> > _loudness;
  Sink<std::vector<std::vector<Real> > > _loudnessBandRatio;
  Source<std::vector<std::vector<Real> > > _beatogram;

 public:
  Beatogram() {
    declareAlgorithm("Beatogram");
    declareInput(_loudness, TOKEN, "loudness");
    declareInput(_loudnessBandRatio, TOKEN, "loudnessBandRatio");
    declareOutput(_beatogram, TOKEN, "beatogram");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BEATOGRAM_H
