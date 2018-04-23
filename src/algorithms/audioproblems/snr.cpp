/*
 * Copyright (C) 2006-2018  Music Technology Group - Universitat Pompeu Fabra
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

#include "snr.h"

using namespace essentia;
using namespace standard;

const char* SNR::name = "SNR";
const char* SNR::category = "Audio Problems";
const char* SNR::description = DOC("");

void SNR::compute() {
  const std::vector<Real>& frame = _frame.get();
  std::vector<Real>& snrPrior = _SNRprior.get();
  Real& snrAverage = _SNRAverage.get();
  Real& snrAverageEma = _SNRAverageEMA.get();

  if (frame.size() != _frameSize){
    _frameSize = frame.size();
    _specSize = _frameSize / 2 + 1;
    E_INFO("SNR: Now input frame size is " << _frameSize << 
    "resizing buffers.");

    _prevY.resize(_frameSize / 2 + 1);
    _noisePsd.resize(_frameSize / 2 + 1);
    snrPrior.resize(_frameSize / 2 + 1);
    _snrInst.resize(_frameSize / 2 + 1);
    _snrPost.resize(_frameSize / 2 + 1);
    _prevSnrPrior.resize(_frameSize / 2 + 1);
    _prevSnrInst.resize(_frameSize / 2 + 1);
    _prevSnrPost.resize(_frameSize / 2 + 1);
    _v.resize(_frameSize / 2 + 1);
    _prevMmse.resize(_frameSize / 2 + 1);

    _windowing->configure("size", _frameSize,
                          "type", "hann",
                          "normalized", false);

    _spectrum->configure("size", _frameSize);
  }
  
  std::vector<Real> windowed;
  _windowing->input("frame").set(frame);
  _windowing->output("frame").set(windowed);
  _windowing->compute();

  std::vector<Real> Y;
  _spectrum->input("frame").set(windowed);
  _spectrum->output("spectrum").set(Y);
  _spectrum->compute();

  if (instantPower(frame) < _noiseThreshold) {
    UpdateNoisePSD(_noisePsd, Y, _alphaNoise);
    SNRPostEst(_snrPost, _noisePsd, Y);
    SNRInstEst(_snrInst, _snrPost);
  }

  else {
    if (sum(_prevSnrPrior) == 0.f)
      for (uint i; i < _specSize; i++)
        _prevSnrPrior[i] = _alphaMmse + (1 - _alphaMmse) 
          * std::max(_prevSnrInst[i], 0.f);

    SNRPostEst(_snrPost, _noisePsd, Y);
    SNRInstEst(_snrInst, _snrPost);

    V(_v, _prevSnrPrior, _prevSnrPost);

    MMSE(_prevMmse, _v, _prevSnrPost, _prevY);

    SNRPriorEst(_alphaMmse, snrPrior, _prevMmse,
                _noisePsd, _snrInst);
    
    for (uint i; i < _specSize; i++)
      _XPsdEst[i] = _noisePsd[i] * snrPrior[i];

    _snrAverage = mean(_XPsdEst) / mean(_noisePsd);

    UpdateEMA(_alphaEma, _snrAverageEma, _snrAverage);

    _prevSnrPrior = snrPrior;

    snrAverageEma = _snrAverageEma;
    snrAverage = _snrAverage;
  }

  _prevNoisePsd = _noisePsd;
  _prevSnrPost = _snrPost;
  _prevSnrInst = _snrInst;
  _prevY = _Y;
}

void SNR::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _frameSize = parameter("frameSize").toInt();
  _noiseThreshold = db2pow(parameter("noiseThreshold").toReal());
  _alphaMmse = parameter("MMSEAlpha").toReal();
  _alphaEma = parameter("MAAlpha").toReal();
  _alphaNoise = parameter("NoiseAlpha").toReal();
  _useBroadbadNoiseCorrection = parameter("useBroadbadNoiseCorrection").toBool();

  _windowing->configure("size", _frameSize,
                        "type", "hann",
                        "normalized", false);

  _spectrum->configure("size", _frameSize);

  _specSize = _frameSize / 2 + 1;
}

void SNR::SNRPriorEst(Real alpha, std::vector<Real> &snrPrior,
                   std::vector<Real> mmse,
                   std::vector<Real> noisePsd,
                   std::vector<Real> snrInst) {
  for (uint i; i < _specSize; i++)
    snrPrior[i] = alpha * pow(mmse[i], 2.f) / noisePsd[i] +
                  (1 - alpha) * std::max(snrInst[i], 0.f);
  };

void SNR::SNRPostEst(std::vector<Real> &snrPost, 
                std::vector<Real> noisePsd,
                std::vector<Real> Y) {
  for (uint i; i < _specSize; i++)
    snrPost[i] = pow(Y[i], 2.f) / noisePsd[i];
};

void SNR::SNRInstEst(std::vector<Real> &snrInst,
                std::vector<Real> snrPost) {
  for (uint i; i < _specSize; i++)
    snrInst[i] = snrPost[i] - 1.f;
};

void SNR::V(std::vector<Real> &v,
        std::vector<Real> snrPrior,
        std::vector<Real> snrPost){
  for (uint i; i < _specSize; i++)
    v[i] = snrPrior[i] / (1.f + snrPrior[i]) * snrPost[i];
};

void SNR::MMSE(std::vector<Real> &mmse,
          std::vector<Real> v,
          std::vector<Real> snrPost,
          std::vector<Real> Y) {
  float g = 0.8862269254527579; // gamma(1.5)

  for (uint i; i < _specSize; i++)
    if (v[i] > 10.f)
      mmse[i] = v[i] * Y[i] / snrPost[i];
    else
      mmse[i] = g * (sqrt(v[i]) / snrPost[i]) * 
                exp(-v[i] / 2.f) * 
                ((1 + v[i]) * iv(0.f, v[i] / 2.f) +
                v[i] * iv(1.f, v[i] / 2.f)) * Y[i];
};

void SNR::UpdateNoisePSD(std::vector<Real> &noisePsd,
                    std::vector<Real> noise,
                    Real alpha) {
    for (uint i; i < _specSize; i++)
      noisePsd[i] = alpha * noisePsd[i] + 
                    (1 - alpha) * pow(noise[i], 2.f);
};

void SNR::UpdateEMA(Real alpha, Real &ema, Real y) {
  ema = alpha * ema + (1 - alpha) * y;
};
