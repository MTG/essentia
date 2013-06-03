/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_SBIC_H
#define ESSENTIA_SBIC_H

#include "algorithm.h"
#include "tnt/tnt.h"

namespace essentia {
namespace standard {

class SBic : public Algorithm {

 private:
  Input<TNT::Array2D<Real> > _features;
  Output<std::vector<Real> > _segmentation;

  int _size1;
  int _size2;
  int _inc1;
  int _inc2;
  Real _cpw;
  int _minLength;
  Real _cp; // complexity penalty

 public:
  SBic() {
    declareInput(_features, "features", "extracted features matrix (rows represent features, and columns represent frames of audio)");
    declareOutput(_segmentation, "segmentation", "a list of frame indices that indicate where a segment of audio begins/ends (the indices of the first and last frame are also added to the list at the beginning and end, respectively)");
  }

  ~SBic() {}

  void declareParameters() {
    declareParameter("size1", "first pass window size [frames]", "[1,inf)", 300);
    declareParameter("inc1", "first pass increment [frames]", "[1,inf)", 60);
    declareParameter("size2", "second pass window size [frames]", "[1,inf)", 200);
    declareParameter("inc2", "second pass increment [frames]", "[1,inf)", 20);
    declareParameter("cpw", "complexity penalty weight", "[0,inf)", 1.5);
    declareParameter("minLength", "minimum length of a segment [frames]", "[1,inf)", 10);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

 private:
  Real logDet(const TNT::Array2D<Real>& matrix) const;
  int bicChangeSearch(const TNT::Array2D<Real>& matrix, int inc, int current) const;
  Real delta_bic(const TNT::Array2D<Real>& matrix, Real segPoint) const;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class SBic : public StreamingAlgorithmWrapper {

 protected:
  Sink<TNT::Array2D<Real> > _features;
  Source<std::vector<Real> > _segmentation;

 public:
  SBic() {
    declareAlgorithm("SBic");
    declareInput(_features, TOKEN, "features");
    declareOutput(_segmentation, TOKEN, "segmentation");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SBIC_H
