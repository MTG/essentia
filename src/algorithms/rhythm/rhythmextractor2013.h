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

#ifndef RHYTHMEXTRACTOR2013_H
#define RHYTHMEXTRACTOR2013_H

#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "algorithm.h"
#include "network.h"

namespace essentia {
namespace streaming {

class RhythmExtractor2013 : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _signal;

  Source<std::vector<Real> > _ticks;
  Source<Real> _confidence;
  Source<Real> _bpm;
  Source<std::vector<Real> > _estimates;
  //Source<std::vector<Real> > _rubatoStart;
  //Source<std::vector<Real> > _rubatoStop;
  //Source<int> _rubatoNumber;
  Source<std::vector<Real> > _bpmIntervals;

  Pool _pool;
  Real _sampleRate;
  Real _periodTolerance;

  int _preferredBufferSize;

  Algorithm* _beatTracker;
  standard::Algorithm* _bpmRubato;
  scheduler::Network* _network;

  std::string _method;
  bool _configured;
  void createInnerNetwork();
  void clearAlgos();

 public:
  RhythmExtractor2013();
  ~RhythmExtractor2013();

  void declareParameters() {
    //declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    // TODO only 44100 sample rate is supported
    declareParameter("method", "the method used for beat tracking", "{multifeature,degara}", "multifeature");
    declareParameter("maxTempo", "the fastest tempo to detect [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "the slowest tempo to detect [bpm]", "[40,180]", 40);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_beatTracker));
    declareProcessStep(SingleShot(this));
  }

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "vectorinput.h"

namespace essentia {
namespace standard {

class RhythmExtractor2013 : public Algorithm {
 protected:
  Input<std::vector<Real> > _signal;
  Output<Real> _bpm;
  Output<std::vector<Real> > _ticks;
  Output<Real> _confidence;
  Output<std::vector<Real> > _estimates;
  //Output<std::vector<Real> > _rubatoStart;
  //Output<std::vector<Real> > _rubatoStop;
  //Output<int> _rubatoNumber;
  Output<std::vector<Real> > _bpmIntervals;

  bool _configured;

  streaming::Algorithm* _rhythmExtractor;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

 public:

  RhythmExtractor2013();
  ~RhythmExtractor2013();

  void declareParameters() {
    //declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("maxTempo", "the fastest tempo to detect [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "the slowest tempo to detect [bpm]", "[40,180]", 40);
    declareParameter("method", "the method used for beat tracking", "{multifeature,degara}", "multifeature");
  }

  void configure();
  void compute();
  void createInnerNetwork();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace standard
} // namespace essentia

#endif // RHYTHMEXTRACTOR2013_H
