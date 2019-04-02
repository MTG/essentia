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

#include "snr.h"

using namespace essentia;
using namespace standard;

const char* SNR::name = "SNR";
const char* SNR::category = "Audio Problems";
const char* SNR::description = DOC("This algorithm computes the SNR of the input audio in a frame-wise manner. The algorithm assumes that:\n"
  "  1. The noise is gaussian.\n"
  "  2. There is a region of noise (without signal) at the beginning of the stream in order to estimate the PSD of the noise.[1]\n"
  "Once the noise PSD is estimated, the algorithm relies on the Ephraim-Malah [2] recursion to estimate the SNR for each frequency bin.\n"
  "The algorithm also returns an overall (a single value for the whole spectrum) SNR estimation and an averaged overall SNR estimation using"
  " Exponential Moving Average filtering.\n"
  "This algorithm throws a Warning if less than 15 frames are used to estimte the noise PSD.\n"
  "\n"
  "References:\n"
  "  [1] Vaseghi, S. V. (2008). Advanced digital signal processing and noise reduction. John Wiley & Sons. Page 336.\n"
  "  [2] Ephraim, Y., & Malah, D. (1984). Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator. "
  "IEEE Transactions on acoustics, speech, and signal processing, 32(6), 1109-1121.\n");


void SNR::compute() {
  const std::vector<Real>& frame = _frame.get();
  std::vector<Real>& snrPrior = _SNRprior.get();
  Real& snrAverage = _SNRAverage.get();
  Real& snrAverageEma = _SNRAverageEMA.get();

  // Check that the buffers match the input size.
  if (frame.size() != _frameSize) {
    _frameSize = frame.size();
    _specSize = _frameSize / 2 + 1;
    E_INFO("SNR: New input frame size is " << _frameSize << 
    "resizing buffers.");

    reset();
  }

  snrPrior.assign(_specSize, 0.f);

  std::vector<Real> windowed;
  _windowing->input("frame").set(frame);
  _windowing->output("frame").set(windowed);
  _windowing->compute();

  std::vector<Real> Y;
  _spectrum->input("frame").set(windowed);
  _spectrum->output("spectrum").set(Y);
  _spectrum->compute();

  // If the frame is under the signal threshold it
  // is used to estimate the noise PSD.
  if (instantPower(frame) < _noiseThreshold) {
    UpdateNoisePSD(_noisePsd, Y, _alphaNoise);
    SNRPostEst(_snrPost, _noisePsd, Y);
    SNRInstEst(_snrInst, _snrPost);
    _counter++;
  }
  else {
    // If the noise PSD haven't been updated at least once,
    // SNR = inf (i.e. signal without noise case).
    if (_counter == 0) {
      snrAverageEma = INFINITY;
      snrAverage = INFINITY;

      _prevNoisePsd = _noisePsd;
      _prevSnrPost = _snrPost;
      _prevSnrInst = _snrInst;
      _prevY = Y;

      return;
    }

    if ((_counter < 15) && (!_warned)) {
      E_WARNING("SNR: Noise PSD was stimated on just " << _counter << 
      " frames. Maybe the audio stream does not have enoguh noise or the"
      " threshold parameter 'noiseThreshold' is not properly set.");

      _warned = true;
      }

    // Initialize SNR a priori.
    if (sum(_prevSnrPrior) == 0.f) {
      for (uint i = 0; i < _specSize; i++)
        _prevSnrPrior[i] = _alphaMmse + (1 - _alphaMmse) *
                           std::max(_prevSnrInst[i], 0.f);
      
      // Check that there are not 0-valued bins to prevent division by 0.
      for (uint i = 0; i < _specSize; i++)
        if (_noisePsd[i] == 0.f)
          _noisePsd[i] += _eps;
    }

    // Steps of the  a priori SNR estimation.
    SNRPostEst(_snrPost, _noisePsd, Y);

    SNRInstEst(_snrInst, _snrPost);

    V(_v, _prevSnrPrior, _prevSnrPost);

    MMSE(_prevMmse, _v, _prevSnrPost, _prevY);

    SNRPriorEst(_alphaMmse, snrPrior, _prevMmse,
                _noisePsd, _snrInst);
    
    // As the SNR a priori is computed in a frequency-wise manner, we need
    // to  estimate the PSD of the signal in order to compute the overall SNR.
    for (uint i = 0; i < _specSize; i++)
      _XPsdEst[i] = _noisePsd[i] * snrPrior[i];

    _snrAverage = mean(_XPsdEst) / mean(_noisePsd);

    // An Exponenitial Moving Average filtered version is also returned.
    UpdateEMA(_alphaEma, _snrAverageEma, _snrAverage);

    _prevSnrPrior = snrPrior;
  }

  snrAverageEma = 10.f * log10(_snrAverageEma);
  snrAverage = 10.f * log10(_snrAverage);

  if (_useBroadbadNoiseCorrection) {
    snrAverageEma -= 10.f * log10(_sampleRate / 2.);
    snrAverage -= 10.f * log10(_sampleRate / 2.);
  }

  _prevNoisePsd = _noisePsd;
  _prevSnrPost = _snrPost;
  _prevSnrInst = _snrInst;
  _prevY = Y;
}


void SNR::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _frameSize = parameter("frameSize").toInt();
  _noiseThreshold = db2pow(parameter("noiseThreshold").toReal());
  _alphaMmse = parameter("MMSEAlpha").toReal();
  _alphaEma = parameter("MAAlpha").toReal();
  _alphaNoise = parameter("NoiseAlpha").toReal();
  _useBroadbadNoiseCorrection = parameter("useBroadbadNoiseCorrection").toBool();

  _specSize = _frameSize / 2 + 1;

  reset();
}


void SNR::SNRPriorEst(Real alpha, std::vector<Real> &snrPrior,
                      std::vector<Real> mmse,
                      std::vector<Real> noisePsd,
                      std::vector<Real> snrInst) {
  for (uint i = 0; i < _specSize; i++) {
    snrPrior[i] = alpha * pow(mmse[i], 2.f) / noisePsd[i] +
                  (1 - alpha) * std::max(snrInst[i], 0.f);
    if (snrPrior[i] == 0.f)
      snrPrior[i] += _eps;
    }
  };


void SNR::SNRPostEst(std::vector<Real> &snrPost, 
                std::vector<Real> noisePsd,
                std::vector<Real> Y) {
  for (uint i = 0; i < _specSize; i++) {
    snrPost[i] = pow(Y[i], 2.f) / noisePsd[i];
    if (snrPost[i] == 0.f)
      snrPost[i] += _eps;
    }
};


void SNR::SNRInstEst(std::vector<Real> &snrInst,
                std::vector<Real> snrPost) {
  for (uint i = 0; i < _specSize; i++)
    snrInst[i] = snrPost[i] - 1.f;
};


void SNR::V(std::vector<Real> &v,
            std::vector<Real> snrPrior,
            std::vector<Real> snrPost){
  for (uint i = 0; i < _specSize; i++)
    v[i] = snrPrior[i] / (1.f + snrPrior[i]) * snrPost[i];
};


void SNR::MMSE(std::vector<Real> &mmse,
               std::vector<Real> v,
               std::vector<Real> snrPost,
               std::vector<Real> Y) {
  static float g = 0.8862269254527579;  // gamma(1.5)

  for (uint i = 0; i < _specSize; i++) {
    if (v[i] > 10.f) {
      mmse[i] = v[i] * Y[i] / snrPost[i];
    }
    else {
      mmse[i] = g * (sqrt(v[i]) / snrPost[i]) * 
                exp(-v[i] / 2.f) * 
                ((1 + v[i]) * cephes::iv(0.f, v[i] / 2.f) +
                v[i] * cephes::iv(1.f, v[i] / 2.f)) * Y[i];
    }
  }
};


void SNR::UpdateNoisePSD(std::vector<Real> &noisePsd,
                    std::vector<Real> noise,
                    Real alpha) {
    for ( uint i = 0; i < _specSize; i++)
      noisePsd[i] = alpha * noisePsd[i] + 
                    (1 - alpha) * pow(noise[i], 2.f);
};


void SNR::UpdateEMA(Real alpha, Real &ema, Real y) {
  ema = alpha * ema + (1 - alpha) * y;
};


void SNR::reset() {
  _prevY.assign(_specSize, 0.f);
  _noisePsd.assign(_specSize, 0.f);
  
  _snrInst.assign(_specSize, 0.f);
  _snrPost.assign(_specSize, 0.f);
  _prevSnrPrior.assign(_specSize, 0.f);
  _prevSnrInst.assign(_specSize, 0.f);
  _prevSnrPost.assign(_specSize, 0.f);
  _v.assign(_specSize, 0.f);
  _prevMmse.assign(_specSize, 0.f);
  _XPsdEst.assign(_specSize, 0.f);

  _snrAverageEma = 0.f;

  _windowing->configure("size", _frameSize,
                        "type", "hann",
                        "normalized", false);

  _spectrum->configure("size", _frameSize);

  _warned = false;
  
  _counter = 0; 
}
