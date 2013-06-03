/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_DANCEABILITY_H
#define ESSENTIA_DANCEABILITY_H

#include "algorithm.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {

class Danceability : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<Real> _danceability;

 public:
  Danceability() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_danceability, "danceability", "the danceability value. Normal values range from 0 to ~3. The higher, the more danceable.");
  }

  void declareParameters() {
    declareParameter("minTau", "minimum segment length to consider [ms]", "(0,inf)", 310.);
    declareParameter("maxTau", "maximum segment length to consider [ms]", "(0,inf)", 8800.);
    declareParameter("tauMultiplier", "multiplier to increment from min to max tau", "[1,inf)", 1.1);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

 protected:
  std::vector<int> _tau;

  Real stddev(const std::vector<Real>& array, int start, int end) const;

  // inline version
  /**
   * from http://mathworld.wolfram.com/LeastSquaresFitting.html
   * instead of "manually" calculating the least squares error by subtracting
   * 'y' and linear_fit(y) we calculate it via the direct formula
   * which uses ssxx, ssxy and ssyy
   **/
  inline Real residualError(const std::vector<Real>& array, int start, int end) const {

    int size = end - start;

    Real mean_x = (size - 1.0) * 0.5;
    Real mean_y = mean(array, start, end);

    Real ssxx = 0.0;
    Real ssyy = 0.0;
    Real ssxy = 0.0;
    Real dx, dy;

    int i = 0;

#define addError(i) dx = (Real)(i) - mean_x;\
                    dy = array[(i)+start] - mean_y;\
                    ssxx += dx * dx;\
                    ssxy += dx * dy;\
                    ssyy += dy * dy

    for (; i<size-8; i+=8) {
      addError(i);
      addError(i+1);
      addError(i+2);
      addError(i+3);
      addError(i+4);
      addError(i+5);
      addError(i+6);
      addError(i+7);
    }

    for (; i<size; i++) {
      addError(i);
    }

#undef addError

    return (ssyy - ssxy * ssxy / ssxx) / size;
  }


};

} // namespace standard
} // namespace essentia


#endif // ESSENTIA_DANCEABILITY_H
