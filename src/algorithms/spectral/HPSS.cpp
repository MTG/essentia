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
#include "essentiamath.h"
#include <essentia/algorithmfactory.h>

using namespace std;
using namespace essentia;
using namespace standard;

const char* HPSS::name = "HPSS";
const char* HPSS::category = "Spectral";
const char* HPSS::description = DOC("Harmonic Percussive Source Separation based on Median Filtering");

void HPSS::configure() {

  // Get parameters.
  _harmonicKernel = parameter("harmonicKernel").toInt();
  _percussiveKernel = parameter("percussiveKernel").toInt();
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

//  std::transform(harmonicSpectrum.begin(), harmonicSpectrum.end(), harmonicSpectrum.begin(), [](Real& c){return c/2.f;});

  _enhancedPercussiveSpectrum = const_spectrum;
  _enhancedHarmonicSpectrum = std::vector<Real> (const_spectrum.size(), 0.0000000001);

  computingEnhancedSpectrograms(const_spectrum, _enhancedPercussiveSpectrum, _enhancedHarmonicSpectrum);

  harmonicSpectrum = _enhancedHarmonicSpectrum;
  percussiveSpectrum = _enhancedPercussiveSpectrum;

//  hardMasking(_enhancedHarmonicSpectrum, _enhancedPercussiveSpectrum, _harmonicMask);
//  hardMasking(_enhancedPercussiveSpectrum, _enhancedHarmonicSpectrum, _percussiveMask);

//  softMasking(_enhancedHarmonicSpectrum, _enhancedPercussiveSpectrum, _harmonicMask);
//  softMasking(_enhancedPercussiveSpectrum, _enhancedHarmonicSpectrum, _percussiveMask);

//  harmonicSpectrum = _harmonicMask;
//  percussiveSpectrum = _percussiveMask;
}

void HPSS::initialize() { }

void HPSS::reset() {
  initialize();
}

void HPSS::hardMasking(const std::vector<Real>& mX, const std::vector<Real>& mX_ref, std::vector<Real>& mask){
  mask.clear();
//  for (int i=1; i<float(mX.size()); ++i){
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

void HPSS::computingResidual(const std::vector<Real>& mX, const std::vector<Real>& mP,const std::vector<Real>& mH,
        std::vector<Real>& mR) { // TODO: not debugged yet
  mR.clear();
  for (size_t i = 1; i < mX.size(); ++i) {
    mR.push_back(mX[i] - (mH[i] + mP[i]));
  }
}


void HPSS::computingEnhancedSpectrograms(const std::vector<Real>& mX, std::vector<Real>& mP, std::vector<Real>& mH) {
  // TODO: fix the syntax
//  _harmonicKernel, _percussiveKernel;

  // TODO: check syntax to: call the MedianFilter class, set kernelSize
  // TODO: move to the proper method when this calls need to be done
//  MedianFilter _percussiveMedianFilter;
//  _percussiveMedianFilter.kernelSize(_percussiveKernel);

  AlgorithmFactory& factory = standard::AlgorithmFactory::instance();
  Algorithm* _harmonicMedian  = factory.create("Median");

//  blocks = mX.shape[0]
//  bins = mX.shape[1]

//  int shift = (_harmonicKernel / 2); // middle of the buffer

  // Percussive median filter computed using 1 median filter
//  mP[b] = medfilt(buffer[:, shift], win_perc)  // time position centered in the middle of the buffer
//  mP = _percussiveMedianFilter.compute(buffer[shift]);

  vector<Real> _harmonicBuffer;
  Real output_harmonicMedian;

  _harmonicMedian->input("array").set(_harmonicBuffer);
  _harmonicMedian->output("median").set(output_harmonicMedian);

  // todo: debug Median algorithm integration!! Change the input and check whether the output make sense

  // Harmonic median filter computed using the median of one bin all over the blocks

  // TODO: I think I implemented the PERCUSSIVE instead of the HARMONIC: I'm grouping through the vertical/frequency dimension
  // Should be correct. But let's review that!

  // TODO: you need to test this. Prove that this is working as you expected!
  // cleaning the enhanced spectrogram for the new input spectrogram
  mH.clear();
  // for each bin in the spectrogram
  size_t nBins = mX.size();
  for(size_t bin=0; bin<nBins; ++bin) { 
    // we are creating a vector which starts at the same bin and finishes _harmonicKernel bins later
    _harmonicBuffer = std::vector<Real>(mX.begin()+bin, mX.begin()+bin+_harmonicKernel);
    // we compute the median for the aforementioned vector
    _harmonicMedian->compute();
    // we store the median in the enhanced spectrogram
    mH.push_back(output_harmonicMedian);
  }

  
std::cout << ("Computed harmonic median filter for one spectrum");

//  mH[b] = np.median(buffer, axis = 1)

  // get median for each bins


  //TODO: check syntax of the buffer updating
//  currentIndex = (currentIndex+1 % _percussiveMedianFilter)
//  buffer[currentIndex] = mX // change the values of the vector inside the vector of vector
//  buffer = np.roll(buffer, -1, axis=1)
//  buffer[:, win_harm-1] = mX[(b+win_harm) % blocks]

//  TODO: what about the time shift? are we giving a delayed answer?
//  time shifting: center the output of the median filter with the middle of the buffer
//  NB: mP[r] could be substituted with per[(b+shift) % blocks] to get the same result avoiding the final shifting!
//  mH = np.roll(mH, shift, 0)
//  mP = np.roll(mP, shift, 0)
}