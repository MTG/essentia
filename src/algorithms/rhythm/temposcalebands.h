/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_TEMPOSCALE_H
#define ESSENTIA_TEMPOSCALE_H

#include "algorithm.h"
#include "essentiautil.h"

namespace essentia {
namespace standard {

class TempoScaleBands : public Algorithm {

 private:
  Input< std::vector<Real> > _bands;
  Output<std::vector<Real> > _scaledBands;
  Output<Real> _cumulBands;

  Real _frameFactor;
  std::vector<Real> _scratchBands;
  std::vector<Real> _oldBands;
  std::vector<Real> _bandsGain;

 public:
  TempoScaleBands() {
    declareInput(_bands, "bands", "the audio power spectrum divided into bands");
    declareOutput(_scaledBands, "scaledBands", "the output bands after scaling");
    declareOutput(_cumulBands, "cumulativeBands", "cumulative sum of the output bands before scaling");
  }

  ~TempoScaleBands() {};

  void declareParameters() {
    declareParameter("frameTime", "the frame rate in samples", "(0,inf)", 512.0);
    Real bandGains[] = {2.0, 3.0, 2.0, 1.0, 1.2, 2.0, 3.0, 2.5};
    declareParameter("bandsGain", "gain for each bands", "", arrayToVector<Real>(bandGains));
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

  Real scale(const Real& value, const Real& c1, const Real& c2, const Real& pwr);

  void reset();

}; // class TempoScaleBands

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TempoScaleBands : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _bands;
  Source<Real> _cumulBands;
  Source<std::vector<Real> > _scaledBands;

 public:
  TempoScaleBands() {
    declareAlgorithm("TempoScaleBands");
    declareInput(_bands, TOKEN, "bands");
    declareOutput(_scaledBands, TOKEN, "scaledBands");
    declareOutput(_cumulBands, TOKEN, "cumulativeBands");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_TEMPOTAP_H
