/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_BPMUBATO_H
#define ESSENTIA_BPMRUBATO_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class BpmRubato : public Algorithm {

 private:
  // input array of features
  Input<std::vector<Real> > _beats;
  // output vector for start timestamps
  Output<std::vector<Real> > _rubatoStart;
  // output vector for stop timestamps
  Output<std::vector<Real> > _rubatoStop;
  Real _tolerance; // variation in % of the current bpm
  Real _shortestRegion;
  Real _longestRegion;
  Real _rubatoOn; // whether or not we are in a rubato region

 public:
  BpmRubato() {
    declareInput(_beats, "beats", "list of detected beat ticks [s]");
    declareOutput(_rubatoStart, "rubatoStart", "list of timestamps where the start of a rubato region was detected [s]");
    declareOutput(_rubatoStop, "rubatoStop", "list of timestamps where the end of a rubato region was detected [s]");
  }

  ~BpmRubato() {};

  void declareParameters() {
    declareParameter("tolerance", "minimum tempo deviation to look for", "[0,1]", 0.08);
    declareParameter("longRegionsPruningTime", "time for the longest constant tempo region inside a rubato region [s]", "[0,inf)", 20.);
    declareParameter("shortRegionsMergingTime", "time for the shortest constant tempo region from one tempo region to another [s]", "[0,inf)", 4.);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

}; // class BpmRubato

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class BpmRubato : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _beats;
  Source<std::vector<Real> > _rubatoStart;
  Source<std::vector<Real> > _rubatoStop;

 public:
  BpmRubato() {
    declareAlgorithm("BpmRubato");
    declareInput(_beats, TOKEN, "beats");
    declareOutput(_rubatoStart, TOKEN, "rubatoStart");
    declareOutput(_rubatoStop, TOKEN, "rubatoStop");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TEMPOTAP_H
