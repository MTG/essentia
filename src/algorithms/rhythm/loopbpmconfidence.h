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

#ifndef ESSENTIA_LOOP_BPM_CONFIDENCE_H
#define ESSENTIA_LOOP_BPM_CONFIDENCE_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {
class LoopBpmConfidence : public Algorithm {

  protected:
    Input<std::vector<Real> > _signal;
    Input<Real> _bpmEstimate;
    Output<Real> _confidence;
    Algorithm* _envelope;


  public:
    LoopBpmConfidence() {
      declareInput(_signal, "signal", "loop audio signal");
      declareInput(_bpmEstimate, "bpmEstimate", "estimated BPM for the audio signal");
      declareOutput(_confidence, "confidence", "confidence value for the BPM estimation");
      _envelope = AlgorithmFactory::create("Envelope");
    }

    ~LoopBpmConfidence(){
    }

    void declareParameters() {
      declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
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

class LoopBpmConfidence : public StreamingAlgorithmWrapper {

 protected:
  Sink<Real> _signal;
  Sink<Real> _bpmEstimate;
  Source<Real> _confidence;

 public:
  LoopBpmConfidence() {
    declareAlgorithm("LoopBpmConfidence");
    declareInput(_signal, TOKEN, "signal");
    declareInput(_bpmEstimate, TOKEN, "bpmEstimate");
    declareOutput(_confidence, TOKEN, "confidence");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_LOOP_BPM_CONFIDENCE_H
