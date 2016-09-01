/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_PERCIVALEVALUATEPULSETRAINS_H
#define ESSENTIA_PERCIVALEVALUATEPULSETRAINS_H

#include "algorithm.h"

namespace essentia {
namespace standard {
class PercivalEvaluatePulseTrains : public Algorithm {

  protected:
    Input<std::vector<Real> > _oss;
    Input<std::vector<Real> > _peakPositions;
    Output<Real> _lag;

    void calculatePulseTrains(const std::vector<Real>& ossWindow,
                              const int lag,
                              Real& magScore,
                              Real& varScore);

  public:
    PercivalEvaluatePulseTrains() {
    declareInput(_oss, "oss", "onset strength signal (or other novelty curve)");
    declareInput(_peakPositions, "positions", "peak positions of BPM candidates");
    declareOutput(_lag, "lag", "best tempo lag estimate");
    }
    ~PercivalEvaluatePulseTrains(){
    }

    void declareParameters() {
    }

    void configure();
    void compute();
    void reset() {}

    static const char* name;
    static const char* category;
    static const char* description;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class PercivalEvaluatePulseTrains : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _oss;
  Sink<std::vector<Real> > _peakPositions;
  Source<Real> _lag;

 public:
  PercivalEvaluatePulseTrains() {
    declareAlgorithm("PercivalEvaluatePulseTrains");
    declareInput(_oss, TOKEN, "oss");
    declareInput(_peakPositions, TOKEN, "positions");
    declareOutput(_lag, TOKEN, "lag");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PERCIVALEVALUATEPULSETRAINS_H
