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

#ifndef ESSENTIA_BPMHISTOGRAM_H
#define ESSENTIA_BPMHISTOGRAM_H

#include "algorithmfactory.h"
#include "streamingalgorithmcomposite.h"
#include "pool.h"
#include "network.h"

namespace essentia {
namespace streaming {

class BpmHistogram : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _signal;

  Source<Real> _bpm;
  Source<std::vector<Real> > _bpmCandidates;
  Source<std::vector<Real> > _bpmMagnitudes;
  // it has to be a TNT::Array cause Pool doesn't support vector<vector<type> >
  Source<TNT::Array2D<Real> > _tempogram;
  Source<std::vector<Real> > _frameBpms;
  Source<std::vector<Real> > _ticks;
  Source<std::vector<Real> > _ticksMagnitude;
  Source<std::vector<Real> > _sinusoid;

  // inner algos
  Algorithm* _frameCutter;
  Algorithm* _windowing;
  Algorithm* _fft;
  Algorithm* _cart2polar;
  Algorithm* _peakDetection;
  scheduler::Network* _network;

  // parameters:
  Real _binWidth;
  Real _minBpm, _maxBpm;
  Real _frameRate;
  Real _bpmTolerance;
  int _frameSize, _hopSize;
  int _maxPeaks;
  int _preferredBufferSize;
  bool _normalize;
  bool _weightByMagnitude;
  bool _constantTempo;
  Real _meanBpm;


  Pool _pool;
  std::vector<Real> _window;

  void computeBpm();

  // functions for computing ticks:
  void createWindow(int size);
  void createTicks(Real bpm);
  void createSinusoid(std::vector<Real>& sinusoid, Real freq, Real phase, int idx);
  void unwrapPhase(Real& ph, const Real& uwph);
  void postProcessBpms(Real mainBpm, std::vector<Real>& bpms);
  void computeHistogram(std::vector<Real>& bpmPositions, std::vector<Real>& bpmMagnitudes);
  Real deviationWeight(Real x, Real mu, Real sigma);


 public:
  BpmHistogram();
  ~BpmHistogram();

  void declareParameters() {
    declareParameter("frameRate", "the sampling rate of the novelty curve [frame/s]", "[1,inf)", 44100./512.);
    declareParameter("frameSize", "the minimum length to compute the fft [s]", "[1,inf)", 4.0);
    declareParameter("zeroPadding", "zero padding factor to compute the fft [s]", "[0,inf)", 0);
    declareParameter("overlap", "the overlap factor", "(0,inf)", 16);
    declareParameter("windowType", "the window type to be used when computing the fft", "", "hann");
    declareParameter("maxPeaks", "the number of peaks to be considered at each spectrum", "(0,inf]", 50);
    declareParameter("minBpm", "the minimum bpm to consider", "[0,inf)", 30.);
    declareParameter("maxBpm", "the maximum bpm to consider", "(0,inf)", 560.);
    declareParameter("weightByMagnitude", "whether to consider peaks' magnitude when building the histogram", "{true,false}", true);
    declareParameter("constantTempo", "whether to consider constant tempo. Set to true when inducina specific tempo", "{true,false}", false);
    declareParameter("tempoChange", "the minimum length to consider a change in tempo as stable [s]", "[0,inf)", 5.);
    declareParameter("bpm", "bpm to induce a certain tempo tracking. Zero if unknown", "[0,inf)", 0.0);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_frameCutter));
    declareProcessStep(SingleShot(this));
  }

  void configure();
  AlgorithmStatus process();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_BPMHISTOGRAM_H
