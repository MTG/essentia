/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_SLICER_H
#define ESSENTIA_SLICER_H

#include "streamingalgorithm.h"

namespace essentia {
namespace streaming {

class Slicer : public Algorithm {
 protected:
  Sink<Real> _input;
  Source<std::vector<Real> > _output;

  // pair of [ startSample, endSample ]
  std::vector<std::pair<int, int> > _slices;
  int _consumed;
  std::string _timeUnits;
  Real _sampleRate;
  std::vector<Real> _startTimes;
  std::vector<Real> _endTimes;
  int _sliceIdx;

  static const int defaultPreferredSize = 4096;

 public:
  Slicer() : Algorithm() {
    declareInput(_input, defaultPreferredSize, "audio", "the input signal");
    declareOutput(_output, 1, "frame", "the frames of the sliced input signal");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]",
                     "(0,inf)", 44100.0);
    declareParameter("startTimes", "the list of start times for the slices "
                     "you want to extract",
                     "", std::vector<Real>());
    declareParameter("endTimes", "the list of end times for the slices you "
                     "want to extract",
                     "", std::vector<Real>());
    declareParameter("timeUnits", "the units of time of the start and end times",
                     "{samples,seconds}", "seconds");
  }

  void configure();
  void reset();
  AlgorithmStatus process();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "algorithm.h"
#include "vectoroutput.h"
#include "vectorinput.h"
#include "network.h"

namespace essentia {
namespace standard {

// Standard non-streaming algorithm comes after the streaming one as it
// depends on it
class Slicer : public Algorithm {
 protected:
  Input<std::vector<Real> > _audio;
  Output<std::vector<std::vector<Real> > > _output;

  bool _configured;

  streaming::Algorithm* _slicer;
  streaming::VectorOutput<std::vector<Real> >* _storage;
  streaming::VectorInput<Real>* _gen;
  scheduler::Network* _network;

  void createInnerNetwork();

 public:
  Slicer() : _configured(false) {
    declareInput(_audio, "audio", "the input audio signal");
    declareOutput(_output, "frame", "the frames of the sliced input signal");

    createInnerNetwork();
  }

  ~Slicer() { delete _network; }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]",
                     "(0,inf)", 44100.);
    declareParameter("startTimes", "the list of start times for the slices "
                     "you want to extract",
                     "", std::vector<Real>());
    declareParameter("endTimes", "the list of end times for the slices you "
                     "want to extract",
                     "", std::vector<Real>());
    declareParameter("timeUnits", "the units of time of the start and end times",
                     "{samples,seconds}", "seconds");
  }

  void configure();

  void compute();
  void reset() { _network->reset(); }

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // ESSENTIA_SLICER_H
