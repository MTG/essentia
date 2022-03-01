/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_PITCHCREPE_H
#define ESSENTIA_PITCHCREPE_H

#include "algorithmfactory.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {

class PitchCREPE : public Algorithm {

 protected:
  Input<std::vector<Real> > _audio;
  Output<std::vector<Real> > _time;
  Output<std::vector<Real> > _frequency;
  Output<std::vector<Real> > _confidence;
  Output<std::vector<std::vector<Real> > > _activations;

  Algorithm* _tensorflowPredictCREPE;

  bool _viterbi;
  float _hopSize;
  const float _sampleRate = 16000;

  // Cents mapping parameters
  std::vector<Real> _centsMapping;
  const int _nPitches = 360;
  const Real _end = 7180;
  const Real _shift = 1997.3794084376191;
  const Real _delta = _end / (_nPitches - 1);

  std::vector<Real> toLocalAverageCents(std::vector<std::vector<Real> > &activations);

 public:
  PitchCREPE() {
    declareInput(_audio, "audio", "the input audio signal sampled at 16000 Hz");
    declareOutput(_time, "time" , "the timestamps on which the pitch was estimated");
    declareOutput(_frequency, "frequency", "the predicted pitch values in Hz");
    declareOutput(_confidence, "confidence", "the confidence of voice activity, between 0 and 1");
    declareOutput(_activations, "activations", "the raw activation matrix");

    _tensorflowPredictCREPE = AlgorithmFactory::create("TensorflowPredictCREPE");
  }

  ~PitchCREPE() {
    if (_tensorflowPredictCREPE) delete _tensorflowPredictCREPE;
  }

  void declareParameters() {
    declareParameter("graphFilename", "the name of the file from which to load the TensorFlow graph", "", "");
    declareParameter("savedModel", "the name of the TensorFlow SavedModel. Overrides parameter `graphFilename`", "", "");
    declareParameter("input", "the name of the input node in the TensorFlow graph", "", "frames");
    declareParameter("output", "the name of the node from which to retrieve the output tensors", "", "model/classifier/Sigmoid");
    declareParameter("hopSize", "the hop size in milliseconds for running pitch estimation", "(0,inf)", 10.0);
    declareParameter("batchSize", "the batch size for prediction. This allows parallelization when a GPU are available. Set it to -1 or 0 to accumulate all the patches and run a single TensorFlow session at the end", "[-1,inf)", 64);
    // CREPE implements temporal smoothing via Viterbi but it is not applied by default and we will leave it unimplemented for now.
    // declareParameter("viterbi", "whether to use Viterbi decoding for temporal smoothing", "{true,false}", true);
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

class PitchCREPE : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _audio;
  Source<std::vector<Real> > _time;
  Source<std::vector<Real> > _frequency;
  Source<std::vector<Real> > _confidence;
  Source<std::vector<std::vector<Real> > > _activations;

 public:
  PitchCREPE() {
    declareAlgorithm("PitchCREPE");
    declareInput(_audio, TOKEN, "audio");
    declareOutput(_time, TOKEN, "time");
    declareOutput(_frequency, TOKEN, "frequency");
    declareOutput(_confidence, TOKEN, "confidence");
    declareOutput(_activations, TOKEN, "activations");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_PITCHCREPE_H
