/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_BPMHISTOGRAMDESCRIPTORS_H
#define ESSENTIA_BPMHISTOGRAMDESCRIPTORS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class BPMHistogramDescriptors : public Algorithm {

 private:
  Input<std::vector<Real> > _bpmIntervals;

  Output<Real> _firstPeakBPM;
  Output<Real> _firstPeakWeight;
  Output<Real> _firstPeakSpread;
  Output<Real> _secondPeakBPM;
  Output<Real> _secondPeakWeight;
  Output<Real> _secondPeakSpread;

 public:
  BPMHistogramDescriptors() {
    declareInput(_bpmIntervals, "bpmIntervals", "the list of bpm intervals [s]");
    declareOutput(_firstPeakBPM, "firstPeakBPM", "value for the highest peak [bpm]");
    declareOutput(_firstPeakWeight, "firstPeakWeight", "weight of the highest peak");
    declareOutput(_firstPeakSpread, "firstPeakSpread", "spread of the highest peak");
    declareOutput(_secondPeakBPM, "secondPeakBPM", "value for the second highest peak [bpm]");
    declareOutput(_secondPeakWeight, "secondPeakWeight", "weight of the second highest peak");
    declareOutput(_secondPeakSpread, "secondPeakSpread", "spread of the second highest peak");
  }

  ~BPMHistogramDescriptors() {};

  void declareParameters() {}

  void compute();

  static const char* name;
  static const char* description;

  static const int maxBPM;
  static const int numPeaks;
  static const int weightWidth;
  static const int spreadWidth;
};

} // namespace standard
} // namespace essentia


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class BPMHistogramDescriptors : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _bpmIntervals;

  Source<Real> _firstPeakBPM;
  Source<Real> _firstPeakWeight;
  Source<Real> _firstPeakSpread;
  Source<Real> _secondPeakBPM;
  Source<Real> _secondPeakWeight;
  Source<Real> _secondPeakSpread;

 public:
  BPMHistogramDescriptors() {
    declareAlgorithm("BPMHistogramDescriptors");
    declareInput(_bpmIntervals, TOKEN, 1, "bpmIntervals");
    declareOutput(_firstPeakBPM, TOKEN, 1, "firstPeakBPM");
    declareOutput(_firstPeakWeight, TOKEN, 1, "firstPeakWeight");
    declareOutput(_firstPeakSpread, TOKEN, 1, "firstPeakSpread");
    declareOutput(_secondPeakBPM, TOKEN, 1, "secondPeakBPM");
    declareOutput(_secondPeakWeight, TOKEN, 1, "secondPeakWeight");
    declareOutput(_secondPeakSpread, TOKEN, 1, "secondPeakSpread");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BPMHISTOGRAMDESCRIPTORS_H
