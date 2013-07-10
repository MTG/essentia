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

#ifndef ESSENTIA_STRONGDECAY_H
#define ESSENTIA_STRONGDECAY_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class StrongDecay : public Algorithm {

 private:
  Input<std::vector<Real> > _signal;
  Output<Real> _strongDecay;
  Algorithm* _centroid;
  Algorithm* _abs;

 public:
  StrongDecay() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_strongDecay, "strongDecay", "the strong decay");

    _centroid = AlgorithmFactory::create("Centroid");
    _abs = AlgorithmFactory::create("UnaryOperator", "type", "abs");
  }

  ~StrongDecay() {
    delete _centroid;
    delete _abs;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void compute();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia


#include "accumulatoralgorithm.h"

namespace essentia {
namespace streaming {

class StrongDecay : public AccumulatorAlgorithm {

 protected:
  Sink<Real> _signal;
  Source<Real> _strongDecay;

  double _centroid;
  double _energy;
  double _weights;
  int _idx;

 public:
  StrongDecay() {
    declareInputStream(_signal, "signal", "the input audio signal");
    declareOutputResult(_strongDecay, "strongDecay", "the strong decay");
    reset();
  }

  void reset();
  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
  }

  void consume();
  void finalProduce();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_STRONGDECAY_H
