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


void Brightness::computeFramesSpectrumPower(const vector<Real>& signal, vector<Real>& framesSpectrumPower, uint nFrames) {
  vector<Real> frame;
  
  framesSpectrumPower.clear();
  framesSpectrumPower.reserve(nFrames);

  _frameCutter->reset();  // Needs to be reset before processing a new signal
  _frameCutter->input("signal").set(signal);
  _frameCutter->output("frame").set(frame);

  for (uint nFrame = 0; nFrame < nFrames; ++nFrame) {  
    _frameCutter->compute();
    
    vector<Real> windowedFrame;
    vector<Real> spectrum;

    _windowing->input("frame").set(frame);
    _windowing->output("frame").set(windowedFrame);
    _windowing->compute();

    _powerSpectrum->input("signal").set(windowedFrame);  // These 
    _powerSpectrum->output("powerSpectrum").set(spectrum);
    _powerSpectrum->compute();

    Real powerSum = 0.0;
    for (const auto& bin : spectrum) {
      powerSum += bin;
    }
    framesSpectrumPower.push_back(powerSum);
  }
}

void Brightness::applyFilterNPasses(Algorithm* filter, const vector<Real>& inputSignal, vector<Real>& outputSignal, int nPasses) {
  vector<Real> currentInput = inputSignal;
  vector<Real> currentOutput;
  for (int i = 0; i < nPasses; ++i) {
    filter->input("signal").set(currentInput);
    filter->output("signal").set(currentOutput);
    filter->compute();
    currentInput = currentOutput;  // For the next pass
  }
  outputSignal = currentOutput;  // Final output after n passes
}


void Brightness::configure() {
  const Real samplingRate = parameter("samplingRate").toReal();
  const int windowSize = parameter("windowSize").toInt();
  const int hopSize = windowSize / 4;  
  const Real minFreq = parameter("minFreq").toReal();
  const Real centroidCrossover = parameter("centroidCrossover").toReal();
  const Real ratioCrossover = parameter("ratioCrossover").toReal();
  
  _minFreqHighPass->configure("cutoffFrequency", minFreq, "sampleRate", samplingRate);
  _centroidCrossoverHighPass->configure("cutoffFrequency", centroidCrossover, "sampleRate", samplingRate);
  _ratioCrossoverHighPass->configure("cutoffFrequency", ratioCrossover, "sampleRate", samplingRate);
  _windowing->configure("type", "hamming", "size", windowSize);
  _frameCutter->configure("frameSize", windowSize, "hopSize", hopSize, "startFromZero", true);
  _powerSpectrum->configure("size", windowSize);
  _centroid->configure("range", samplingRate / 2.0);
}

void Brightness::computeFramesCentroids(const vector<Real>& signal, vector<Real>& centroids, uint nFrames) {
  vector<Real> frame;
  
  centroids.clear();
  centroids.reserve(nFrames);

  _frameCutter->reset();  // Needs to be reset before processing a new signal
  _frameCutter->input("signal").set(signal);
  _frameCutter->output("frame").set(frame);

  for (uint nFrame = 0; nFrame < nFrames; ++nFrame) {  
    _frameCutter->compute();
    
    vector<Real> windowedFrame;
    vector<Real> spectrum;

    _windowing->input("frame").set(frame);
    _windowing->output("frame").set(windowedFrame);
    _windowing->compute();

    _powerSpectrum->input("signal").set(windowedFrame);
    _powerSpectrum->output("powerSpectrum").set(spectrum);
    _powerSpectrum->compute();

    Real centroid = 0.0;
    _centroid->input("array").set(spectrum);
    _centroid->output("centroid").set(centroid);
    _centroid->compute();
    
    centroids.push_back(centroid);
  }
}

void Brightness::compute() {
  const vector<Real>& signal = _signal.get();
  Real& brightness = _brightness.get();

  // TODO: this algorithm expects a mono signal of the sampling rate specified in the parameters. Should we check for that and throw an exception if not?

  // Loudness normalization step: the original algorithm has a loudness normalization step which we skip because, according to our evaluation,
  // it does not lead to significant changes in the results

  // Create 3 filtered versions of the original signal using given cutoff frequencies
  int nPasses = 3; // We apply it multiple times to better approximate original implementation's filter response
  vector<Real> signalMinFreq;
  vector<Real> signalCentroid;
  vector<Real> signalRatio;
  applyFilterNPasses(_minFreqHighPass, signal, signalMinFreq, nPasses);
  applyFilterNPasses(_centroidCrossoverHighPass, signalMinFreq, signalCentroid, nPasses);
  applyFilterNPasses(_ratioCrossoverHighPass, signalMinFreq, signalRatio, nPasses);
  
  // Normalize the signals according to maximum absolute value for signalMinFreq
  Real maxAbs = 0.0;
  for (const auto& sample : signalMinFreq) { maxAbs = std::max(maxAbs, std::abs(sample)); }
  if (maxAbs == 0.0) { brightness = 0.0; return; }  // Avoid division by zero, and if the signal is completely silent, brightness is 0
  for (auto& sample : signalMinFreq) { sample /= maxAbs; }
  for (auto& sample : signalCentroid) { sample /= maxAbs; }
  for (auto& sample : signalRatio) { sample /= maxAbs; }

  // Calculate frame-by-frame power spectrum sum for each of the 3 signals. Will be used later for estimating brightness
  uint nFrames = (signalMinFreq.size() - _frameCutter->parameter("frameSize").toInt()) / _frameCutter->parameter("hopSize").toInt() + 1;
  vector<Real> signalMinFreqFramesSpectrumPower;
  vector<Real> signalCentroidFramesSpectrumPower;
  vector<Real> signalRatioFramesSpectrumPower;
  computeFramesSpectrumPower(signalMinFreq, signalMinFreqFramesSpectrumPower, nFrames);
  computeFramesSpectrumPower(signalCentroid, signalCentroidFramesSpectrumPower, nFrames);
  computeFramesSpectrumPower(signalRatio, signalRatioFramesSpectrumPower, nFrames);

  // Caculate frame-by-frame spectral centroids for the centroid-filtered signal. Will be used later for estimating brightness
  vector<Real> signalCentroidCentroids;
  computeFramesCentroids(signalCentroid, signalCentroidCentroids, nFrames);
  
  // Compute R, weighted average of the ratio values
  Real sumRatios = 0.0;  // sum of ratios
  Real sumWeights = 0.0;  // sum of weights
  for (uint t = 0; t < nFrames; ++t) {
    Real P_all = signalMinFreqFramesSpectrumPower[t];
    Real P_ratio = signalRatioFramesSpectrumPower[t];
    if (P_all > 0.0) {
      Real ratio_t = P_ratio / P_all;
      sumRatios += ratio_t * P_all;
      sumWeights += P_all;
    }
  }
  Real R = (sumWeights > 0.0) ? sumRatios / sumWeights : 0.0;

  // Compute C, weighted average of the centroid values
  Real sumCentroids = 0.0;  // sum of centroids
  sumWeights = 0.0;  // sum of weights
  for (uint t = 0; t < nFrames; ++t) {
    Real P_cent = signalCentroidFramesSpectrumPower[t];
    if (P_cent > 0.0) {
      sumCentroids += signalCentroidCentroids[t] * P_cent;
      sumWeights += P_cent;
    }
  }
  Real C = (sumWeights > 0.0) ? sumCentroids / sumWeights : 0.0;

  // Apply linear model to get brightness from R and C
  brightness = 4.6131280180 * log10(R) + 17.3788893093 * log10(C) + 17.4347337506;

  // Clip signal to be in range [0, 100] and set output
  if (brightness < 0.0) {
    brightness = 0.0;
  } else if (brightness > 100.0) {
    brightness = 100.0;
  }
}
