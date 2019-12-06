/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include "HPSS.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* HPSS::name = "HPSS";
const char* HPSS::category = "Spectral";
const char* HPSS::description = DOC("Harmonic Percussive Source Separation based on Median Filtering");

void HPSS::configure() {

  // Get parameters.
  _harmonicKernel = parameter("harmonicKernel").toInt();
  // _percussiveKernel = parameter("percussiveKernel").toInt();
  _harmonicMargin = parameter("harmonicMargin").toReal();
  _percussiveMargin = parameter("percussiveMargin").toReal();
  _power = parameter("power").toReal();
  _frameSize = parameter("frameSize").toInt();
  initialize();
}


void HPSS::compute() {
  const vector<Real>& const_spectrum = _spectrum.get();
  vector<Real>& percussiveSpectrum = _percussiveSpectrum.get();
  vector<Real>& harmonicSpectrum = _harmonicSpectrum.get();
  std::vector<Real> _enhancedPercussiveSpectrum, _enhancedHarmonicSpectrum;
  std::vector<Real> _harmonicMask, _percussiveMask;

  if (const_spectrum.size() <= 1)
    throw EssentiaException("HPSS: input vector is empty");

  if (const_spectrum.size() != _frameSize) {
    E_INFO("HPSS: input spectrum size does not match '_frameSize' "
           "parameter. Reconfiguring the algorithm.");
    _frameSize = const_spectrum.size();
    initialize();
  }

  computingHarmonicEnhancedSpectrogram(const_spectrum, _enhancedHarmonicSpectrum);
  computingPercussiveEnhancedSpectrogram(const_spectrum, _enhancedPercussiveSpectrum);

  harmonicSpectrum = _enhancedHarmonicSpectrum;
  percussiveSpectrum = _enhancedPercussiveSpectrum;

//  hardMasking(_enhancedHarmonicSpectrum, _enhancedPercussiveSpectrum, _harmonicMask);
//  hardMasking(_enhancedPercussiveSpectrum, _enhancedHarmonicSpectrum, _percussiveMask);

//  softMasking(_enhancedHarmonicSpectrum, _enhancedPercussiveSpectrum, _harmonicMask);
//  softMasking(_enhancedPercussiveSpectrum, _enhancedHarmonicSpectrum, _percussiveMask);

//  harmonicSpectrum = _harmonicMask;
//  percussiveSpectrum = _percussiveMask;
}

void HPSS::initialize() {
  _harmonicMatrixBuffer = std::vector<std::vector<Real>> (_frameSize, std::vector<Real>(_harmonicKernel, 0));

  _percussiveMedianFilter->configure(INHERIT("kernelSize"));
}

void HPSS::reset() {
  initialize();
}

void HPSS::hardMasking(const std::vector<Real>& mX, const std::vector<Real>& mX_ref, std::vector<Real>& mask){
  mask.clear();
  for (size_t i=1; i<mX.size(); ++i){
    if (mX[i] > mX_ref[i]){
      mask.push_back(1.f); // TODO: avoid push_back for optimization?
    } else {
      mask.push_back(0.f);
    }
  }
}

void HPSS::softMasking(const std::vector<Real>& mX, const std::vector<Real>& mX_ref, std::vector<Real>& mask){
  mask.clear();
  Real Z, X, X_ref;
  for (size_t i=1; i<mX.size(); ++i) {
    // Rescaling and power application
    Z = max(mX[i], mX_ref[i]);
//    if (Z < std::numeric_limits<Real>::min()) { // invalid index // Todo: check if the max is positive
//      if (_split_zeros) mask.push_back(0.5f);
//      else mask.push_back(0.0f);
//    } else {

//      X = (mX[i] / Z) ^ _power; // Todo: how to compute power(Real, Real)?
//      X_ref = (mX_ref[i] / Z) ^ _power;

      X = (mX[i] / Z);
      X_ref = (mX_ref[i] / Z);

      // Computing mask
      mask.push_back(X / (X + X_ref));
    }
//  }
}

void HPSS::computingResidual(const std::vector<Real>& mX, const std::vector<Real>& mP,const std::vector<Real>& mH, std::vector<Real>& mR) { // TODO: not debugged yet
  mR.clear();
  for (size_t i = 1; i < mX.size(); ++i) {
    mR.push_back(mX[i] - (mH[i] + mP[i]));
  }
}

void HPSS::computingHarmonicEnhancedSpectrogram(const std::vector<Real>& mX, std::vector<Real>& mH){
  // for each bin in the spectrogram 
  mH.clear();
  for(size_t bin=0; bin<_frameSize; ++bin) { 
    std::rotate(_harmonicMatrixBuffer[bin].begin(),_harmonicMatrixBuffer[bin].begin()+1,_harmonicMatrixBuffer[bin].begin()+_harmonicKernel);
    // old version, which one is more efficient?
    // for(size_t backInTime=0; backInTime<(_harmonicKernel-1); ++backInTime){
    //   _harmonicMatrixBuffer[bin][backInTime] = _harmonicMatrixBuffer[bin][backInTime+1];
    // }

    _harmonicMatrixBuffer[bin][_harmonicKernel-1] = mX[bin];

    _harmonicMedian->input("array").set(_harmonicMatrixBuffer[bin]);
    _harmonicMedian->output("median").set(_output_harmonicMedian);
    _harmonicMedian->compute();

    mH.push_back(_output_harmonicMedian);
  }
}

void HPSS::computingPercussiveEnhancedSpectrogram(const std::vector<Real>& mX, std::vector<Real>& mP){
  _percussiveMedianFilter->input("array").set(mX);
  _percussiveMedianFilter->output("filteredArray").set(mP);
  _percussiveMedianFilter->compute();
}