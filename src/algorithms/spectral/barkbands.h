/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_BARKBANDS_H
#define ESSENTIA_BARKBANDS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class BarkBands : public Algorithm {

 private:
  Input<std::vector<Real> > _spectrumInput;
  Output<std::vector<Real> > _bandsOutput;

  Algorithm* _freqBands;

 public:
  BarkBands() {
    declareInput(_spectrumInput, "spectrum", "the input spectrum");
    declareOutput(_bandsOutput, "bands", "the energy of the bark bands");
    _freqBands = AlgorithmFactory::create("FrequencyBands");
  }

  ~BarkBands() {
    if (_freqBands) delete _freqBands;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "[0,inf)", 44100.);
    declareParameter("numberBands", "the number of desired barkbands", "[1,28]", 27);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

} // namespace essentia
} // namespace standard

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class BarkBands : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _spectrumInput;
  Source<std::vector<Real> > _bandsOutput;

 public:
  BarkBands() {
    declareAlgorithm("BarkBands");
    declareInput(_spectrumInput, TOKEN, "spectrum");
    declareOutput(_bandsOutput, TOKEN, "bands");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_BARKBANDS_H
