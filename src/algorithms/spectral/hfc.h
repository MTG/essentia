/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_HFC_H
#define ESSENTIA_HFC_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class HFC : public Algorithm {

 protected:
  Input<std::vector<Real> > _spectrum;
  Output<Real> _hfc;

  std::string _type;
  Real _sampleRate;

 public:
  HFC() {
    declareInput(_spectrum, "spectrum", "the input audio spectrum");
    declareOutput(_hfc, "hfc", "the high-frequency coefficient");
  }

  void declareParameters() {
    declareParameter("type", "the type of HFC coefficient to be computed", "{Masri,Jensen,Brossier}", "Masri");
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf]", 44100.0);
  }

  void configure();
  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class HFC : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrum;
  Source<Real> _hfc;

 public:
  HFC() {
    declareAlgorithm("HFC");
    declareInput(_spectrum, TOKEN, "spectrum");
    declareOutput(_hfc, TOKEN, "hfc");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_HFC_H
