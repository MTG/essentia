/*
 * Copyright (C) 2006-2019  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_SNR_H
#define ESSENTIA_SNR_H

#include "essentiamath.h"
#include "algorithm.h"
#include "algorithmfactory.h"
#include "3rdparty/cephes/bessel/bessel.h"

namespace essentia {
namespace standard {

class SNR : public Algorithm {
 protected:
  Input<std::vector<Real> > _frame;
  Output<std::vector<Real> > _SNRprior;
  Output<Real> _SNRAverage;
  Output<Real> _SNRAverageEMA;

  void SNRPriorEst(Real alpha, std::vector<Real> &snrPrior,
                   std::vector<Real> mmse,
                   std::vector<Real> noisePsd,
                   std::vector<Real> snrInst);

  void SNRPostEst(std::vector<Real> &snrPost, 
                  std::vector<Real> noisePsd,
                  std::vector<Real> Y);

  void SNRInstEst(std::vector<Real> &snrInst,
                  std::vector<Real> snrPost);

  void V(std::vector<Real> &v,
         std::vector<Real> snrPrior,
         std::vector<Real> snrPost);

  void MMSE(std::vector<Real> &mmse,
            std::vector<Real> v,
            std::vector<Real> snrPost,
            std::vector<Real> Y);

  void UpdateNoisePSD(std::vector<Real> &noisePsd,
                      std::vector<Real> noise,
                      Real alpha);

  void UpdateEMA(Real alpha, Real &ema, Real y);

  void reset();

  Real _sampleRate;
  Real _noiseThreshold;
  Real _alphaMmse;
  Real _alphaEma;
  Real _alphaNoise;
  bool _useBroadbadNoiseCorrection;
  bool _warned;
  uint _frameSize;
  uint _specSize;
  uint _counter;

  std::vector<Real> _Y;
  std::vector<Real> _noisePsd;
  std::vector<Real> _snrPrior;
  std::vector<Real> _snrInst;
  std::vector<Real> _snrPost;
  std::vector<Real> _XPsdEst;

  std::vector<Real> _prevY;
  std::vector<Real> _prevNoisePsd;
  std::vector<Real> _prevSnrPrior;
  std::vector<Real> _prevSnrInst;
  std::vector<Real> _prevSnrPost;
  std::vector<Real> _v;
  std::vector<Real> _prevMmse;

  Real _snrAverage;
  Real _snrAverageEma;

  Algorithm* _windowing;
  Algorithm* _spectrum;

  Real _eps = std::numeric_limits<Real>::epsilon();

 public:
  SNR() {
    declareInput(_frame, "frame", "the input audio frame");
    declareOutput(_SNRAverage, "instantSNR", "SNR value for the the current frame");
    declareOutput(_SNRAverageEMA, "averagedSNR", "averaged SNR through an Exponential Moving Average filter");
    declareOutput(_SNRprior, "spectralSNR", "instant SNR for each frequency bin");
 
   _windowing = AlgorithmFactory::create("Windowing");
   _spectrum  = AlgorithmFactory::create("Spectrum");
  
  }

  ~SNR() {
    if (_windowing) delete _windowing;
    if (_spectrum) delete _spectrum;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("frameSize", "the size of the input frame", "(1,inf)", 512);
    declareParameter("noiseThreshold", "Threshold to detect frames without signal", "(-inf,0]", -40.);
    declareParameter("MMSEAlpha", "Alpha coefficient for the MMSE estimation [1].", "[0,1]", 0.98);
    declareParameter("MAAlpha", "Alpha coefficient for the EMA SNR estimation [2]", "[0,1]", 0.95);
    declareParameter("NoiseAlpha", "Alpha coefficient for the EMA noise estimation [2]", "[0,1]", 0.9);
    declareParameter("useBroadbadNoiseCorrection", "flag to apply the -10 * log10(BW) broadband noise correction factor", "{true,false}", true);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;
};

} // namespace essentia
} // namespace standard


#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class SNR : public StreamingAlgorithmWrapper {
 protected:
  Sink<std::vector<Real> > _frame;
  Source<std::vector<Real> > _SNRprior;
  Source<Real> _SNRAverage;
  Source<Real> _SNRAverageEMA;

 public:
  SNR() {
    declareAlgorithm("SNR");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_SNRAverage, TOKEN, "instantSNR");
    declareOutput(_SNRAverageEMA, TOKEN, "averagedSNR");
    declareOutput(_SNRprior, TOKEN, "spectralSNR");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SNR_H
