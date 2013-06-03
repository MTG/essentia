/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_TRIMMER_H
#define ESSENTIA_TRIMMER_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Trimmer : public Algorithm {

 private:
  Input<std::vector<Real> > _input;
  Output<std::vector<Real> > _output;

  long long _startIndex;
  long long _endIndex;

 public:
  Trimmer() {
    declareInput(_input, "signal", "the input signal");
    declareOutput(_output, "signal", "the trimmed signal");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the input audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("startTime", "the start time of the slice you want to extract [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice you want to extract [s]", "[0,inf)", 1.0e6);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

}// namespace standard
}// namespace essentia


#include "streamingalgorithm.h"


namespace essentia {
namespace streaming {

class Trimmer : public Algorithm {
 protected:
  Sink<Real> _input;
  Source<Real> _output;

  int _preferredSize;
  long long _startIndex;
  long long _endIndex;
  long long _consumed;

  static const int defaultPreferredSize = 4096;

 public:
  Trimmer() : Algorithm(), _preferredSize(defaultPreferredSize) {
    declareInput(_input, _preferredSize, "signal", "the input signal");
    declareOutput(_output, _preferredSize, "signal", "the trimmed signal");

    _output.setBufferType(BufferUsage::forAudioStream);
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the input audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("startTime", "the start time of the slice you want to extract [s]", "[0,inf)", 0.0);
    declareParameter("endTime", "the end time of the slice you want to extract [s]", "[0,inf)", 1.0e6);
  }

  void configure();
  AlgorithmStatus process();

  void reset() {
    Algorithm::reset();
    _consumed = 0;
  }

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_TRIMMER_H
