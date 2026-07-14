/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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

#include "brightness.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Brightness::name = "Brightness";
const char* Brightness::category = "Timbre";
const char* Brightness::description = DOC("This algorithm computes the brightness of the analyzed audio in a scale from [0-100]. A bright sound is one that is clear/vibrant and/or contains significant high-pitched elements.\n"
"\n"
"References:\n"
"  [1] A. Pearce, S. Safavi, T. Brookes, R. Mason, W. Wang, and M. Plumbley, \"AudioCommons Timbral Models,\" Github Repository, "
"  https://github.com/AudioCommons/timbral_models.\n\n"
"  [2] A. Pearce, S. Safavi, T. Brookes, R. Mason, W. Wang, and M. Plumbley, \"D5.8: Release of timbral characterisation "
"  tools for semantically annotating non-musical content\", January 2019. https://audiocommons.github.io/materials/.\n");


void Brightness::computeFrameSpectrumEnergies(const vector<Real>& signal, vector<Real>& energies, uint nFrames) {
  // TODO: not sure if we are actually computing energies here, as we're simply summing.
  // Should this be renamed?
  vector<Real> frame;
  vector<Real> spectrum;

  energies.clear();
  energies.reserve(nFrames);
  _frameCutter->reset();
  _frameCutter->input("signal").set(signal);
  _frameCutter->output("frame").set(frame);

  for (uint nFrame = 0; nFrame < nFrames; ++nFrame) {  
    _frameCutter->compute();
    _spectrum->input("frame").set(frame);
    _spectrum->output("spectrum").set(spectrum);
    _spectrum->compute();

    Real energy = 0.0;
    for (const auto& bin : spectrum) {
      energy += bin; // * bin;
    }
    energies.push_back(energy);
  }
}


void Brightness::configure() {
  const Real minFreq = parameter("minFreq").toReal();
  const Real ratioCrossover = parameter("ratioCrossover").toReal();
  const Real centroidCrossover = parameter("centroidCrossover").toReal();

  _minFreqHighPass->configure("cutoffFrequency", minFreq);
  _centroidCrossoverHighPass->configure("cutoffFrequency", centroidCrossover);
  _ratioCrossoverHighPass->configure("cutoffFrequency", ratioCrossover);

  const int windowSize = parameter("windowSize").toInt();
  // TODO Make sure this returns int
  const int hopSize = 3 * windowSize / 4;  
  _frameCutter->configure("frameSize", windowSize, "hopSize", hopSize);
  _spectrum->configure("size", windowSize);
  
}

void Brightness::compute() {
  const vector<Real>& signal = _signal.get();
  Real& brightness = _brightness.get();

  // 1) Apply _minFreqHighPass to signal 3 times
  vector<Real> signalMinFreq = signal;  // Make a copy so we don't modify the original (?)
  for (int i = 0; i < 3; ++i) {
    _minFreqHighPass->input("signal").set(signalMinFreq);
    _minFreqHighPass->output("signal").set(signalMinFreq);
    _minFreqHighPass->compute();
  }

  // 2) Apply _centroidCrossoverHighPass to signalMinFreq 3 times to get signalCentroid
  vector<Real> signalCentroid = signalMinFreq;  // Make a copy so we don't modify the original (?)
  for (int i = 0; i < 3; ++i) {
    _centroidCrossoverHighPass->input("signal").set(signalCentroid);
    _centroidCrossoverHighPass->output("signal").set(signalCentroid);
    _centroidCrossoverHighPass->compute();
  }

  // 3) Apply _ratioCrossoverHighPass to signalMinFreq 3 times to get signalRatio
  vector<Real> signalRatio = signalMinFreq;  // Make a copy so we don't modify the original (?)
  for (int i = 0; i < 3; ++i) {
    _ratioCrossoverHighPass->input("signal").set(signalRatio);
    _ratioCrossoverHighPass->output("signal").set(signalRatio);
    _ratioCrossoverHighPass->compute();
  }

  // 4) Noramlize the signals according to maximum absolute value for signalMinFreq
  // 4.1) Get maximum absolute value of signalMinFreq. TODO: is there a better way to do this?
  Real maxAbs = 0.0;
  for (const auto& sample : signalMinFreq) {
    maxAbs = std::max(maxAbs, std::abs(sample));
  }
  if (maxAbs == 0.0) {
    // Silence signal, set brightness to 0 and return
    // TODO: should this be checked before filtering?
    brightness = 0.0;
    return;
  }

  // 4.2) Normalize signalMinFreq, signalCentroid, and signalRatio by maxAbs
  _scaler->configure("factor", 1.0 / maxAbs); // TODO: is it fine to configure things in the compute function?
  _scaler->input("signal").set(signalMinFreq);
  _scaler->output("signal").set(signalMinFreq);
  _scaler->compute();
  _scaler->input("signal").set(signalCentroid);
  _scaler->output("signal").set(signalCentroid);
  _scaler->compute();
  _scaler->input("signal").set(signalRatio);
  _scaler->output("signal").set(signalRatio);
  _scaler->compute();

  // 5) Split the signal in frames and save spectrum energy for each
  vector<Real> signalMinFreqEnergies;
  vector<Real> signalCentroidEnergies;
  vector<Real> signalRatioEnergies;
  uint nFrames = signalMinFreq.size() / _frameCutter->parameter("hopSize").toInt();  // TODO: is this correct? what if the last frame is not complete?
  
  // 5.1) Get energies for signalMinFreq
  computeFrameSpectrumEnergies(signalMinFreq, signalMinFreqEnergies, nFrames);

  // 5.2) Get energies for signalCentroid
  computeFrameSpectrumEnergies(signalCentroid, signalCentroidEnergies, nFrames);

  // 5.3) Get energies for signalRatio
  computeFrameSpectrumEnergies(signalRatio, signalRatioEnergies, nFrames);

  // 6) Weighted average of the ratio values
  Real sumRatios = 0.0;  // sum of ratios
  Real sumWeights = 0.0;  // sum of weights
  for (uint t = 0; t < nFrames; ++t) {
    Real P_all = signalMinFreqEnergies[t];
    Real P_ratio = signalRatioEnergies[t];
    if (P_all > 0.0) {
      Real ratio_t = P_ratio / P_all;
      sumRatios += ratio_t * P_all;
      sumWeights += P_all;
    }
  }
  Real R = (sumWeights > 0.0) ? sumRatios / sumWeights : 0.0;

  // 7) Weighted average of the centroid values
  // 7.1) Get the centroid for each frame of signalCentroid
  vector<Real> signalCentroidCentroids;
  vector<Real> frame;
  signalCentroidCentroids.reserve(nFrames);
  _frameCutter->reset();
  _frameCutter->input("signal").set(signalCentroid);
  _frameCutter->output("frame").set(frame);
  for (uint nFrame = 0; nFrame < nFrames; ++nFrame) {
    Real centroid = 0.0;
    _frameCutter->compute();
    _centroid->input("array").set(frame);
    _centroid->output("centroid").set(centroid);
    _centroid->compute();
    signalCentroidCentroids.push_back(centroid);
  }
  // 7.2) Compute the weighted average of the centroid values
  Real sumCentroids = 0.0;  // sum of centroids
  sumWeights = 0.0;  // sum of weights
  for (uint t = 0; t < nFrames; ++t) {
    Real P_cent = signalCentroidEnergies[t];
    if (P_cent > 0.0) {
      sumCentroids += signalCentroidCentroids[t] * P_cent;
      sumWeights += P_cent;
    }
  }
  Real C = (sumWeights > 0.0) ? sumCentroids / sumWeights : 0.0;

  // DEBUG: print log10(R) and log10(C)
  std::cout << "log10(R): " << log10(R) << ", log10(C): " << log10(C) << std::endl;

  // 8) Apply linear model to get brightness from R and C
  /*
  # Linear model
  features = [log10(R), log10(C), 1]
  coeff    = [17.3788893093, 17.4347337506, 4.6131280180]
  brightness = dot(features, coeff)
  */
  brightness = 17.3788893093 * log10(R) + 17.4347337506 * log10(C) + 4.6131280180;

  // 9) Clip signal to be in range [0, 100] and set output
  if (brightness < 0.0) {
    brightness = 0.0;
  } else if (brightness > 100.0) {
    brightness = 100.0;
  }
}
