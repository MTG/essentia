#ifndef RHYTHM_DESCRIPTORS_H
#define RHYTHM_DESCRIPTORS_H

#include "streamingalgorithmcomposite.h"
#include "algorithm.h"
#include "pool.h"
#include "vectorinput.h"
#include "network.h"

namespace essentia {
namespace streaming {

class RhythmDescriptors : public AlgorithmComposite {
 protected:
  Algorithm* _bpmHistogramDescriptors;
  Algorithm* _rhythmExtractor;

  // from RhythmExtractor
  SinkProxy<Real> _signal;

  Source<Real> _bpm;
  Source<std::vector<Real> > _ticks;
  Source<std::vector<Real> > _estimates;
  Source<std::vector<Real> > _rubatoStart;
  Source<std::vector<Real> > _rubatoStop;
  Source<std::vector<Real> > _bpmIntervals;

  // from BPMHistogramDescriptors
  SourceProxy<Real> _firstPeakBPM;
  SourceProxy<Real> _firstPeakWeight;
  SourceProxy<Real> _firstPeakSpread;
  SourceProxy<Real> _secondPeakBPM;
  SourceProxy<Real> _secondPeakWeight;
  SourceProxy<Real> _secondPeakSpread;
  
  scheduler::Network* _network;
  Pool _pool;
  bool _configured;

 public:
  RhythmDescriptors();
  ~RhythmDescriptors();

  void declareParameters() {}

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_rhythmExtractor)); 
    declareProcessStep(SingleShot(this));
  }

  void createInnerNetwork();
  void clearAlgos();
  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

class RhythmDescriptors : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<Real> _bpm;
  Output<std::vector<Real> > _ticks;
  Output<std::vector<Real> > _estimates;
  Output<std::vector<Real> > _bpmIntervals;
  Output<std::vector<Real> > _rubatoStart;
  Output<std::vector<Real> > _rubatoStop;
  
  Output<Real> _firstPeakBPM;
  Output<Real> _firstPeakSpread;
  Output<Real> _firstPeakWeight;
  Output<Real> _secondPeakBPM;
  Output<Real> _secondPeakSpread;
  Output<Real> _secondPeakWeight;
  
  bool _configured;

  streaming::Algorithm* _rhythmDescriptors;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  RhythmDescriptors();
  ~RhythmDescriptors();

  void declareParameters() {}

  void configure();
  void createInnerNetwork();
  void compute();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif
