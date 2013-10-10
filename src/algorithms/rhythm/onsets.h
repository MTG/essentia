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

#ifndef ESSENTIA_ONSETS_H
#define ESSENTIA_ONSETS_H

#include "algorithmfactory.h"
#include "tnt/tnt.h"

namespace essentia {
namespace standard {

class Onsets : public Algorithm {

 protected:
  Input<TNT::Array2D<Real> > _detections;
  Input<std::vector<Real> > _weights;
  Output<std::vector<Real> > _onsets;

  Real _frameRate;
  int _bufferSize;
  Real _silenceThreshold;
  Real _alpha;

  Algorithm* _movingAverage;

 public:
  Onsets() {
    declareInput(_detections, "detections", "matrix containing onset detection functions--rows represent the values of different detection functions and columns represent different frames of audio (i.e. detections[i][j] represents the value of the ith detection function for the jth frame of audio)");
    declareInput(_weights, "weights", "the weighting coefficicients for each detection function, must be the same as the first dimension of \"detections\"");
    declareOutput(_onsets, "onsets", "the onset times [s]");

    _movingAverage = AlgorithmFactory::create("MovingAverage");
  }

  ~Onsets() {
    delete _movingAverage;
  }

  void declareParameters() {
    declareParameter("frameRate", "frames per second", "(0,inf)", 44100.0/512.0);
    declareParameter("alpha", "the proportion of the mean included to reject smaller peaks--filters very short onsets", "[0,1]", 0.1);
    declareParameter("delay", "the number of frames used to compute the threshold--size of short-onset filter", "(0,inf)", 5);
    declareParameter("silenceThreshold", "the threshold for silence", "[0,1]", 0.020);
  }

  void configure();
  void compute();
  void reset() {
    _movingAverage->reset();
  }

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Onsets : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _weights;
  Sink<TNT::Array2D<Real> > _detections;
  Source<std::vector<Real> > _onsets;

 public:
  Onsets() {
    declareAlgorithm("Onsets");
    declareInput(_detections, TOKEN, "detections");
    declareInput(_weights, TOKEN, "weights");
    declareOutput(_onsets, TOKEN, "onsets");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_ONSETS_H
