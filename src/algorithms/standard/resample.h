/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_RESAMPLE_H
#define ESSENTIA_RESAMPLE_H

#include <samplerate.h>
#include "algorithm.h"

namespace essentia {
namespace standard {

class Resample : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _resampled;

 public:
  Resample() {
    declareInput(_signal, "signal", "the input signal");
    declareOutput(_resampled, "signal", "the resampled signal");
  }

  void declareParameters() {
    declareParameter("inputSampleRate", "the sampling rate of the input signal [Hz]", "(0,inf)", 44100.);
    declareParameter("outputSampleRate", "the sampling rate of the output signal [Hz]", "(0,inf)", 44100.);
    declareParameter("quality", "the quality of the conversion, 0 for best quality", "[0,4]", 1);
  }

  void configure();

  void compute();

  static const char* name;
  static const char* description;

 protected:
  double _factor;
  int _quality;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class Resample : public Algorithm {

 protected:
  Sink<Real> _signal;
  Source<Real> _resampled;
  int _preferredSize;

  SRC_STATE* _state;
  SRC_DATA _data;
  int _errorCode;
  float _delay;

 public:
  Resample() : _state(0) {
    _preferredSize = 4096; // arbitrary
    declareInput(_signal, _preferredSize, "signal", "the input signal");
    declareOutput(_resampled, _preferredSize, "signal", "the resampled signal");

    // useless as we do it anyway in the configure() method
    //_resampled.setBufferType(BufferUsage::forAudioStream);
  }

  ~Resample();

  void declareParameters() {
    declareParameter("inputSampleRate", "the sampling rate of the input signal [Hz]", "(0,inf)", 44100.);
    declareParameter("outputSampleRate", "the sampling rate of the output signal [Hz]", "(0,inf)", 44100.);
    declareParameter("quality", "the quality of the conversion, 0 for best quality", "[0,4]", 1);
  }

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_RESAMPLE_H
