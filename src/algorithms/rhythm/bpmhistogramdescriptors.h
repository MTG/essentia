/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#ifndef ESSENTIA_BPMHISTOGRAMDESCRIPTORS_H
#define ESSENTIA_BPMHISTOGRAMDESCRIPTORS_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class BpmHistogramDescriptors : public Algorithm {

 private:
  Input<std::vector<Real> > _bpmIntervals;

  Output<Real> _firstPeakBPM;
  Output<Real> _firstPeakWeight;
  Output<Real> _firstPeakSpread;
  Output<Real> _secondPeakBPM;
  Output<Real> _secondPeakWeight;
  Output<Real> _secondPeakSpread;

 public:
  BpmHistogramDescriptors() {
    declareInput(_bpmIntervals, "bpmIntervals", "the list of bpm intervals [s]");
    declareOutput(_firstPeakBPM, "firstPeakBPM", "value for the highest peak [bpm]");
    declareOutput(_firstPeakWeight, "firstPeakWeight", "weight of the highest peak");
    declareOutput(_firstPeakSpread, "firstPeakSpread", "spread of the highest peak");
    declareOutput(_secondPeakBPM, "secondPeakBPM", "value for the second highest peak [bpm]");
    declareOutput(_secondPeakWeight, "secondPeakWeight", "weight of the second highest peak");
    declareOutput(_secondPeakSpread, "secondPeakSpread", "spread of the second highest peak");
  }

  ~BpmHistogramDescriptors() {};

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

class BpmHistogramDescriptors : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _bpmIntervals;

  Source<Real> _firstPeakBPM;
  Source<Real> _firstPeakWeight;
  Source<Real> _firstPeakSpread;
  Source<Real> _secondPeakBPM;
  Source<Real> _secondPeakWeight;
  Source<Real> _secondPeakSpread;

 public:
  BpmHistogramDescriptors() {
    declareAlgorithm("BpmHistogramDescriptors");
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
