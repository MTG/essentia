/*
 * Copyright (C) 2006-2020  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_TEMPOCNN_H
#define ESSENTIA_TEMPOCNN_H

#include "algorithmfactory.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {

class TempoCNN : public Algorithm {

 protected:
  Input<std::vector<Real> > _audio;
  Output<Real> _globalTempo;
  Output<std::vector<Real> > _localTempo;
  Output<std::vector<Real> > _localTempoProbs;

  Algorithm* _tensorflowPredictTempoCNN;

  std::string _aggregationMethod;

  std::vector<std::vector<Real> > _predictions;

 public:
  TempoCNN() {
    declareInput(_audio, "audio", "the input audio signal sampled at 11025 Hz");
    declareOutput(_globalTempo, "globalTempo" , "the overall tempo estimation in BPM");
    declareOutput(_localTempo, "localTempo", "the patch-wise tempo estimations in BPM");
    declareOutput(_localTempoProbs, "localTempoProbabilities", "the patch-wise tempo probabilities");

    _tensorflowPredictTempoCNN = AlgorithmFactory::create("TensorflowPredictTempoCNN");
  }

  ~TempoCNN() {
    if (_tensorflowPredictTempoCNN) delete _tensorflowPredictTempoCNN;
  }

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file containing the model to use", "", Parameter::STRING);
    declareParameter("input", "the name of the input node in the TensorFlow graph", "", "input");
    declareParameter("output", "the name of the node from which to retrieve the tempo bins activations", "", "output");
    declareParameter("patchHopSize", "the number of frames between the beginnings of adjacent patches. 0 to avoid overlap", "[0,inf)", 128);
    declareParameter("lastPatchMode", "what to do with the last frames: `repeat` them to fill the last patch or `discard` them", "{discard,repeat}", "discard");
    declareParameter("batchSize", "number of patches to process in parallel. Use -1 to accumulate all the patches and run a single TensorFlow session at the end of the stream.", "[-1,inf)", 1);
    declareParameter("aggregationMethod", "method used to estimate the global tempo.", "{majority,mean,median}", "majority");

  }

  void configure();
  void compute();

  static const char* name;
  static const char* category;
  static const char* description;

};

} //namespace standard
} //namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TempoCNN : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _audio;
  Source<Real> _globalTempo;
  Source<std::vector<Real> > _localTempo;
  Source<std::vector<Real> > _localTempoProbs;

 public:
  TempoCNN() {
    declareAlgorithm("TempoCNN");
    declareInput(_audio, TOKEN, "audio");
    declareOutput(_globalTempo, TOKEN, "globalTempo");
    declareOutput(_localTempo, TOKEN, "localTempo");
    declareOutput(_localTempoProbs, TOKEN, "localTempoProbabilities");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TEMPOCNN_H
