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

#ifndef ESSENTIA_TEMPOTAP_H
#define ESSENTIA_TEMPOTAP_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class TempoTap : public Algorithm {

 protected:
  // input array of features
  Input<std::vector<Real> > _featuresFrame;

  // list of phase candidates
  Output<std::vector<Real> > _phases;
  // list of period estimates
  Output<std::vector<Real> > _periods;


 public:
  TempoTap() {
    declareInput(_featuresFrame, "featuresFrame", "input temporal features");
    declareOutput(_periods, "periods", "list of tempo estimates found for each input feature, in frames");
    declareOutput(_phases, "phases", "list of initial phase candidates found for each input feature, in frames");

    _autocorr = AlgorithmFactory::create("AutoCorrelation");
    _peakDetector = AlgorithmFactory::create("PeakDetection");
  }

  ~TempoTap() {
    delete _autocorr;
    delete _peakDetector;
  };

  void declareParameters() {
    declareParameter("frameSize", "number of audio samples in a frame", "(0,inf)", 256);
    declareParameter("numberFrames", "number of feature frames to buffer on", "(0,inf)", 1024);
    declareParameter("frameHop", "number of feature frames separating two evaluations", "(0,inf)", 1024);
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("tempoHints", "optional list of initial beat locations, to favor the detection of pre-determined tempo period and beats alignment [s]", "", std::vector<Real>());
    declareParameter("maxTempo", "fastest tempo allowed to be detected [bpm]", "[60,250]", 208);
    declareParameter("minTempo", "slowest tempo allowed to be detected [bpm]", "[40,180]", 40);
  }

  void reset();
  void configure();
  void compute();

  void computePeriods(const std::vector<std::vector<Real> >& features);
  void computePhases(const std::vector<std::vector<Real> >& features);

  static const char* name;
  static const char* description;



 protected:
  // scope of one column of the _features array
  Real _frameTime;

  // some memory space we need for computations
  std::vector<std::vector<Real> > _acf;
  std::vector<std::vector<Real> > _mcomb;
  std::vector<std::vector<Real> > _phasesOut;

  // the cumulated frame features array
  std::vector<std::vector<Real> > _featuresNew;
  std::vector<std::vector<Real> > _featuresOld;

  // buffer to store peaks and magnitudes
  std::vector<Real> _peaksPositions, _peaksMagnitudes;

  // algorithms
  Algorithm* _autocorr;
  Algorithm* _peakDetector;
  int _numberFrames; // length of features array
  int _frameHop;

  // some constants
  int _nPeaks; // number of peaks to look for in each feature
  int _maxLag; // maximum lag (minimum tempo period)
  int _minLag; // minimum lag (maximum tempo period)

  int _comblen; // length of the multi comb filter
  int _maxelem; // maximum number of elements in the comb
  std::vector<Real> _weighting; // comb output weighting

}; // class TempoTap

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class TempoTap : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _featuresFrame;
  Source<std::vector<Real> > _phases;
  Source<std::vector<Real> > _periods;

 public:
  TempoTap() {
    declareAlgorithm("TempoTap");
    declareInput(_featuresFrame, TOKEN, "featuresFrame");
    declareOutput(_periods, TOKEN, "periods");
    declareOutput(_phases, TOKEN, "phases");
  }
};

} // namespace streaming
} // namespace essentia


#endif // ESSENTIA_TEMPOTAP_H
