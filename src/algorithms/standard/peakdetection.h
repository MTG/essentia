/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_PEAKDETECTION_H
#define ESSENTIA_PEAKDETECTION_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class PeakDetection : public Algorithm {

 protected:
  Input<std::vector<Real> > _array;
  Output<std::vector<Real> > _values;
  Output<std::vector<Real> > _positions;

  Real _minPos;
  Real _maxPos;
  Real _threshold;
  int _maxPeaks;
  Real _range;
  bool _interpolate;
  std::string _orderBy;

 public:
  PeakDetection() {
    declareInput(_array, "array", "the input array");
    declareOutput(_positions, "positions", "the positions of the peaks");
    declareOutput(_values, "amplitudes", "the amplitudes of the peaks");
  }

  void declareParameters() {
    declareParameter("range", "the input range", "(0,inf)", 1.0);
    declareParameter("maxPeaks", "the maximum number of returned peaks", "[1,inf)", 100);
    declareParameter("maxPosition", "the maximum value of the range to evaluate", "(0,inf)", 1.0);
    declareParameter("minPosition", "the minimum value of the range to evaluate", "[0,inf)", 0.0);
    declareParameter("threshold", "peaks below this given threshold are not output", "(-inf,inf)", -1e6);
    declareParameter("orderBy", "the ordering type of the output peaks (ascending by position or descending by value)", "{position,amplitude}", "position");
    declareParameter("interpolate", "boolean flag to enable interpolation", "{true,false}", true);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

private:
  void interpolate(const Real leftVal, const Real middleVal, const Real rightVal, int currentBin, Real& resultVal, Real& resultBin) const;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PeakDetection : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _array;
  Source<std::vector<Real> > _positions;
  Source<std::vector<Real> > _values;

 public:
  PeakDetection() {
    declareAlgorithm("PeakDetection");
    declareInput(_array, TOKEN, "array");
    declareOutput(_positions, TOKEN, "positions");
    declareOutput(_values, TOKEN, "amplitudes");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PEAKDETECTION_H
