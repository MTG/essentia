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

#include "pitchyinprobabilities.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;


const char* PitchYinProbabilities::name = "PitchYinProbabilities";
const char* PitchYinProbabilities::category = "Pitch";
const char* PitchYinProbabilities::description = DOC("This algorithm estimates the fundamental frequencies, their probabilities given the frame of a monophonic music signal. It is a part of the implementation of the probabilistic Yin algorithm [1].\n"
"\n"
"An exception is thrown if an empty signal is provided.\n"
"\n"
"References:\n"
"  [1] M. Mauch and S. Dixon, \"pYIN: A Fundamental Frequency Estimator\n"
"  Using Probabilistic Threshold Distributions,\" in Proceedings of the\n"
"  IEEE International Conference on Acoustics, Speech, and Signal Processing\n"
"  (ICASSP 2014)Project Report, 2004");


void PitchYinProbabilities::configure() {
  _frameSize = parameter("frameSize").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _lowAmp = parameter("lowAmp").toReal();
  _preciseTime = parameter("preciseTime").toBool();

  _yin.resize(_frameSize/2 + 1);
  _peakProb.resize(_yin.size());

  // Pre-processing
  _FFT->configure("negativeFrequencies", true,
                  "size", _frameSize);
  _IFFT->configure("negativeFrequencies", true,
                   "size", _frameSize);
  _RMSALGO->configure();
}

Real PitchYinProbabilities::parabolicInterpolation(const std::vector<Real> yinBuffer, const size_t tau, const size_t yinBufferSize) {
  // this is taken almost literally from Joren Six's Java implementation
  if (tau == yinBufferSize) // not valid anyway.
  {
    return static_cast<Real>(tau);
  }
  
  Real betterTau = 0.0;
  if (tau > 0 && tau < yinBufferSize-1) {
    Real s0, s1, s2;
    s0 = yinBuffer[tau-1];
    s1 = yinBuffer[tau];
    s2 = yinBuffer[tau+1];
      
    Real adjustment = (s2 - s0) / (2 * (2 * s1 - s2 - s0));
      
    if (abs(adjustment)>1) adjustment = 0;
      
    betterTau = tau + adjustment;
  } else {
    betterTau = tau;
  }
  return betterTau;
}

void PitchYinProbabilities::slowDifference(const std::vector<Real> sig, std::vector<Real> &yinBuffer) 
{
  yinBuffer[0] = 0;

  int startPoint = 0;
  int endPoint = 0;
  // Compute difference function
  for (int tau=1; tau < (int) yinBuffer.size(); ++tau) {
    yinBuffer[tau] = 0.;
    startPoint = yinBuffer.size()/2 - tau/2;
    endPoint = startPoint + yinBuffer.size();
    for (int j=startPoint; j < endPoint; ++j) {
      yinBuffer[tau] += pow(sig[j+tau] - sig[j], 2);
    }
  }
}

void PitchYinProbabilities::fastDifference(const std::vector<Real> in, std::vector<Real> &yinBuffer, const size_t yinBufferSize) 
{
    
    // DECLARE AND INITIALISE
    // initialisation of most of the arrays here was done in a separate function,
    // with all the arrays as members of the class... moved them back here.
    
    size_t frameSize = 2 * (yinBufferSize-1);
    
    vector<Real> audioTransformedReal(frameSize, 0.);
    vector<Real> audioTransformedImag(frameSize, 0.);
    vector<Real> kernel(frameSize, 0.);
    vector<Real> kernelTransformedReal(frameSize, 0.);
    vector<Real> kernelTransformedImag(frameSize, 0.);

    vector<Real> yinStyleACFReal(frameSize, 0.);
    vector<Real> yinStyleACFImag(frameSize, 0.);
    vector<Real> powerTerms(yinBufferSize, 0.);
    
    for (size_t j = 0; j < yinBufferSize; ++j)
    {
        yinBuffer[j] = 0.; // set to zero
    }
    
    // POWER TERM CALCULATION
    // ... for the power terms in equation (7) in the Yin paper
    powerTerms[0] = 0.0;
    for (size_t j = 0; j < yinBufferSize; ++j) {
        powerTerms[0] += in[j] * in[j];
    }

    // now iteratively calculate all others (saves a few multiplications)
    for (size_t tau = 1; tau < yinBufferSize; ++tau) {
        powerTerms[tau] = powerTerms[tau-1] - in[tau-1] * in[tau-1] + in[tau+yinBufferSize] * in[tau+yinBufferSize];  
    }

    // YIN-STYLE AUTOCORRELATION via FFT
    // 1. data
    vector<std::complex<Real> > inComplex(frameSize);
    vector<std::complex<Real> > audioTransformed(frameSize);
    for (size_t i=0; i<frameSize; i++) {
      inComplex[i] = std::complex<Real>(in[i], 0.);
    }
    _FFT->input("frame").set(inComplex);
    _FFT->output("fft").set(audioTransformed);
    _FFT->compute();
    // cout << audioTransformed.size() << endl;
    for (size_t i=0; i<frameSize; i++) {
      audioTransformedReal[i] = audioTransformed[i].real();
      audioTransformedImag[i] = audioTransformed[i].imag();
  }
    
    // 2. half of the data, disguised as a convolution kernel
    // yinBufferSize is frameSize/2+1
    for (size_t j = 0; j < yinBufferSize; ++j) {
        kernel[j] = in[yinBufferSize-1-j];
    }
    vector<std::complex<Real> > kernelComplex(frameSize);
    for (size_t i=0; i<frameSize; i++) {
      kernelComplex[i] = std::complex<Real>(kernel[i], 0.);
    }
    vector<std::complex<Real> > kernelTransformed(frameSize);
    _FFT->input("frame").set(kernelComplex);
    _FFT->output("fft").set(kernelTransformed);
    _FFT->compute();
    for (size_t i=0; i<frameSize; i++) {
      kernelTransformedReal[i] = kernelTransformed[i].real();
      kernelTransformedImag[i] = kernelTransformed[i].imag();
    }

    // 3. convolution via complex multiplication -- written into
    vector<std::complex<Real> > yinStyleACF(frameSize);
    for (size_t j = 0; j < frameSize; ++j) {
        yinStyleACFReal[j] = audioTransformedReal[j]*kernelTransformedReal[j] - audioTransformedImag[j]*kernelTransformedImag[j]; // real
        yinStyleACFImag[j] = audioTransformedReal[j]*kernelTransformedImag[j] + audioTransformedImag[j]*kernelTransformedReal[j]; // imaginary
        yinStyleACF[j] = std::complex<Real>(yinStyleACFReal[j], yinStyleACFImag[j]);
    }

    _IFFT->input("frame").set(yinStyleACF);
    _IFFT->output("fft").set(audioTransformed);
    _IFFT->compute();
    for (size_t j = 0; j < frameSize; ++j) {
      audioTransformedReal[j] = audioTransformed[j].real();
      audioTransformedImag[j] = audioTransformed[j].imag();
    }
    
    // CALCULATION OF difference function
    for (size_t j = 0; j < yinBufferSize; ++j) {
        // taking only the real part
        yinBuffer[j] = powerTerms[0] + powerTerms[j] - 2 * audioTransformedReal[j+yinBufferSize-1];
    }
}

void PitchYinProbabilities::compute() {
  const vector<Real>& signal = _signal.get();
  if (signal.empty()) {
    throw EssentiaException("PitchYinProbabilities: Cannot compute pitch detection on empty signal frame.");
  }
  if ((int) signal.size() != _frameSize) {
    Algorithm::configure( "frameSize", int(signal.size()) );
  } 

  vector<Real>& pitch = _pitch.get();
  vector<Real>& probabilities = _probabilities.get();
  Real& RMS = _RMS.get();
  
  if (_preciseTime) {
    slowDifference(signal, _yin);
  } else {
    fastDifference(signal, _yin, size_t(_yin.size()));
  }

  // Compute a cumulative mean normalized difference function
  _yin[0] = 1;
  Real sum = 0.; 
  for (int tau=1; tau < (int) _yin.size(); ++tau) {
    sum += _yin[tau];
    if (sum == 0) {
      _yin[tau] = 1;
    } else {
      _yin[tau] *= tau / sum;
    }

    // Cannot simply check for sum==0 because NaN will be also produced by 
    // infinitely small values 
    // if (isnan(_yin[tau])) {
    //   _yin[tau] = 1;
    // }
  }
  // _yin[tau] is equal to 1 in the case if the df value for 
  // this tau is the same as the mean across all df values from 1 to tau   

  // Calculate YIN probabilities
  // beta distribution used as parameter priors
  static float betaDist2[100] = {0.012614,0.022715,0.030646,0.036712,0.041184,0.044301,0.046277,0.047298,0.047528,0.047110,0.046171,0.044817,0.043144,0.041231,0.039147,0.036950,0.034690,0.032406,0.030133,0.027898,0.025722,0.023624,0.021614,0.019704,0.017900,0.016205,0.014621,0.013148,0.011785,0.010530,0.009377,0.008324,0.007366,0.006497,0.005712,0.005005,0.004372,0.003806,0.003302,0.002855,0.002460,0.002112,0.001806,0.001539,0.001307,0.001105,0.000931,0.000781,0.000652,0.000542,0.000449,0.000370,0.000303,0.000247,0.000201,0.000162,0.000130,0.000104,0.000082,0.000065,0.000051,0.000039,0.000030,0.000023,0.000018,0.000013,0.000010,0.000007,0.000005,0.000004,0.000003,0.000002,0.000001,0.000001,0.000001,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000};

  size_t minTau = 2;
  size_t maxTau = _yin.size();

  Real minWeight = 0.01;
  size_t nThreshold = 100;
  int nThresholdInt = nThreshold;

  vector<Real> thresholds(nThresholdInt);
  vector<Real> distribution(nThresholdInt);
  _peakProb.assign(_yin.size(), 0.0);

  for (int i = 0; i < nThresholdInt; ++i) {
    distribution[i] = betaDist2[i];
    thresholds[i] = 0.01 + i*0.01;
  }

  int currThreshInd = nThreshold-1;
  size_t tau;
  tau = minTau;
  size_t minInd = 0;
  Real minVal = 42.0;
  Real sumProb = 0.0;
  while (tau+1 < maxTau) {
    if (_yin[tau] < thresholds[thresholds.size()-1] && _yin[tau+1] < _yin[tau]) {
      while (tau + 1 < maxTau && _yin[tau+1] < _yin[tau]) {
        tau++;
      }

      if (_yin[tau] < minVal && tau > 2) {
        minVal = _yin[tau];
        minInd = tau;
      }

      currThreshInd = nThresholdInt-1;
      while (thresholds[currThreshInd] > _yin[tau] && currThreshInd > -1) {
        _peakProb[tau] += distribution[currThreshInd];
        currThreshInd--;
      }
      sumProb += _peakProb[tau];
      tau++;
    } else {
      tau++;
    }
  }

  if (_peakProb[minInd] > 1) {
    E_WARNING("WARNING: yin has prob > 1 ??? I'm returning all zeros instead.");
    _peakProb = vector<Real>(_yin.size());
  }
    
  Real nonPeakProb = 1.0;
  if (sumProb > 0) {
    for (size_t i = minTau; i < maxTau; ++i)
    {
        _peakProb[i] = _peakProb[i] / sumProb * _peakProb[minInd];
        nonPeakProb -= _peakProb[i];
    }
  }

  if (minInd > 0) {
    _peakProb[minInd] += nonPeakProb * minWeight;
  }

  // calculate RMS of the signal, use only size of _yin
  vector<Real>::const_iterator beginYin = signal.begin();
  vector<Real>::const_iterator endYin = signal.begin() + _yin.size();
  vector<Real> signalYinSize(beginYin, endYin);
  _RMSALGO->input("array").set(signalYinSize);
  _RMSALGO->output("rms").set(RMS);
  _RMSALGO->compute();

  // reuse the vector
  _freq.resize(0);
  _peakProb_freq.resize(0);

  // calculate frequency probabilities
  for (size_t iBuf = 0; iBuf < _yin.size(); ++iBuf) {
    if (_peakProb[iBuf] > 0) {
      Real currentF0 = _sampleRate * (1.0 / parabolicInterpolation(_yin, iBuf, _yin.size()));
      _freq.push_back(currentF0);
      _peakProb_freq.push_back(_peakProb[iBuf]);
    }
  }

  // convert the frequency to cents
  bool isLowAmplitude = (RMS < _lowAmp);

  for (size_t iCandidate = 0; iCandidate < _freq.size(); ++iCandidate) {
    Real pitchCents = hz2cents(_freq[iCandidate]);
    _freq[iCandidate] = pitchCents;
    if (isLowAmplitude) {
      // lower the probabilities of the frequencies by calculating the weighted sum
      // if the current frame is the low amplitude
      Real factor = ((RMS+0.01 * _lowAmp) / (1.01 * _lowAmp));
      _peakProb_freq[iCandidate] = _peakProb_freq[iCandidate]*factor;
    }
  }

  pitch = _freq;
  probabilities = _peakProb_freq;
}
