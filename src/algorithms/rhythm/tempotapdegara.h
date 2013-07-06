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

#ifndef ESSENTIA_TEMPOTAPDEGARA_H
#define ESSENTIA_TEMPOTAPDEGARA_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class TempoTapDegara : public Algorithm {

 protected:
  Input<std::vector<Real> > _onsetDetections;
  Output<std::vector<Real> > _ticks;

 public:
  TempoTapDegara() {
    declareInput(_onsetDetections, "onsetDetections", "the input frame-wise vector of onset detection values");
    declareOutput(_ticks, "ticks", "the list of resulting ticks [s]");

    _movingAverage = AlgorithmFactory::create("MovingAverage");
    _frameCutter = AlgorithmFactory::create("FrameCutter");
    _autocorrelation = AlgorithmFactory::create("AutoCorrelation");
  }

  ~TempoTapDegara() {
    if (_movingAverage) delete _movingAverage;
    if (_frameCutter) delete _frameCutter;
    if (_autocorrelation) delete _autocorrelation;
  };

  void declareParameters() {
    declareParameter("sampleRateODF", "the sampling rate of the onset detection function [Hz]", "(0,inf)", 44100./512);
    declareParameter("resample", "use upsampling of the onset detection function (may increase accuracy)", "{none,x2,x3,x4}", "none");
    declareParameter("maxTempo", "fastest tempo allowed to be detected [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "slowest tempo allowed to be detected [bpm]", "[40,180]", 40);
  }

  void reset();
  void configure();
  void compute();

  static const char* name;
  static const char* description;

 private:
  // Davies' beat periods estimation:
  int _smoothingWindowHalfSize;
  static const int _numberCombs = 4;
  static const Real _frameDurationODF = 5.944308390022676;
  Real _sampleRateODF;
  int _hopSizeODF;
  Real _hopDurationODF;
  int _resample;
  size_t _numberFramesODF;
  int _periodMinIndex;
  int _periodMaxIndex;
  int _periodMaxUserIndex;
  int _periodMinUserIndex;
  std::vector<Real> _tempoWeights;
  std::vector<std::vector<Real> > _transitionsViterbi;  // transition matrix for Viterbi
  Algorithm* _autocorrelation;
  Algorithm* _movingAverage;
  Algorithm* _frameCutter;
  void createTempoPreferenceCurve();
  void createViterbiTransitionMatrix();
  void findViterbiPath(const std::vector<Real>& prior,
                     const std::vector<std::vector<Real> > transitionMatrix,
                     const std::vector<std::vector<Real> >& observations,
                     std::vector<Real>& path);
  void computeBeatPeriodsDavies(std::vector<Real> detections,
                                std::vector<Real>& beatPeriods,
                                std::vector<Real>& beatEndPositions);
  void adaptiveThreshold(std::vector<Real>& array, int smoothingHalfSize);

  // Degara's beat tracking from periods:
  static const Real _alpha = 0.5; // decoding weighting parameter
  static const Real _sigma_ibi = 0.025; // std of the inter-beat interval pdf,
                                       // models potential variations in the
                                       // inter-beat interval in secs.
  int _numberStates;    // number HMM states
  Real _resolutionODF;  // time resolution of ODF
  size_t _numberFrames; // number of ODF values
  void computeBeatsDegara(std::vector <Real>& detections,
                          const std::vector<Real>& beatPeriods,
                          const std::vector<Real>& beatEndPositions,
                          std::vector<Real>& ticks);
  void computeHMMTransitionMatrix(const std::vector<Real>& ibiPDF,
                                  std::vector<std::vector<Real> >& transitions);
  void decodeBeats(std::map<Real, std::vector<std::vector<Real> > >& transitionMatrix,
                   const std::vector<Real>& beatPeriods,
                   const std::vector<Real>& beatEndPositions,
                   const std::vector<std::vector<Real> >& biy,
                   std::vector<int>& sequenceStates);

  void gaussianPDF(std::vector<Real>& gaussian, Real gaussianStd, Real step, Real scale=1.);
}; // class TempoTapDegara

} // namespace standard
} // namespace essentia


#include "streamingalgorithmcomposite.h"
#include "pool.h"

namespace essentia {
namespace streaming {

class TempoTapDegara : public AlgorithmComposite {

 protected:
  SinkProxy<Real> _onsetDetections;
  Source<Real> _ticks;

  Pool _pool;
  Algorithm* _poolStorage;
  standard::Algorithm * _tempoTapDegara;

 public:
  TempoTapDegara();
  ~TempoTapDegara();

  void declareParameters() {
    declareParameter("sampleRateODF", "the sampling rate of the onset detection function [Hz]", "(0,inf)", 44100./512);
    declareParameter("resample", "use upsampling of the onset detection function (may increase accuracy)", "{none,x2,x3,x4}", "none");
    declareParameter("maxTempo", "fastest tempo allowed to be detected [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "slowest tempo allowed to be detected [bpm]", "[40,180]", 40);
  }

  void configure() {
    _tempoTapDegara->configure(INHERIT("sampleRateODF"),
                               INHERIT("resample"),
                               INHERIT("maxTempo"),
                               INHERIT("minTempo"));
  }

  void declareProcessOrder() {
    declareProcessStep(SingleShot(_poolStorage));
    declareProcessStep(SingleShot(this));
  }

  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_TEMPOTAPDEGARA_H
