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

#include <complex>
#include "tempoestimator.h"
#include "poolstorage.h"
#include "algorithmfactory.h"
#include "essentiamath.h"
#include <essentia/streaming/algorithms/fileoutput.h>

using namespace std;

namespace essentia {
namespace streaming {

const char* TempoEstimator::name = "TempoEstimator";
const char* TempoEstimator::description = DOC("This algorithm estimates the tempo in bpm from an input signal as described in [1]."
"Status: work in progress.\n"
"\n"
"\n"
"References:\n"
"  [1] Percival, G., & Tzanetakis, G. (2014). Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses.\n"
"  IEEE/ACM Transactions on Audio, Speech, and Language Processing, 22(12), 1765–1776.\n\n");


TempoEstimator::TempoEstimator()
  : AlgorithmComposite(), _frameCutter(0), _windowing(0), _spectrum(0), _scaleSpectrum(0),
  _shiftSpectrum(0), _logSpectrum(0), _normSpectrum(0), _flux(0), _lowPass(0),
  _frameCutterOSS(0), _autoCorrelation(0), _enhanceHarmonics(0), _peakDetection(0),
  _evaluatePulseTrains(0), _configured(false) {
  declareInput(_signal, "signal", "input signal");
  declareOutput(_bpm, "bpm", "the tempo estimation [bpm]");
}

void TempoEstimator::createInnerNetwork() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter      = factory.create("FrameCutter");
  _windowing        = factory.create("Windowing");
  _spectrum         = factory.create("Spectrum");
  _scaleSpectrum    = factory.create("UnaryOperator");
  _shiftSpectrum    = factory.create("UnaryOperator");
  _logSpectrum      = factory.create("UnaryOperator");
  _normSpectrum     = factory.create("UnaryOperator");
  _flux             = factory.create("Flux");
  _lowPass          = factory.create("LowPass");
  _frameCutterOSS   = factory.create("FrameCutter");
  _autoCorrelation  = factory.create("AutoCorrelation");
  _enhanceHarmonics = factory.create("EnhanceHarmonics");
  _peakDetection    = factory.create("PeakDetection");
  _evaluatePulseTrains = factory.create("EvaluatePulseTrains");

  // Connect internal algorithms
  // Compute Step 1 of algorithm
  _signal                                     >>  _frameCutter->input("signal");
  _frameCutter->output("frame")               >>  _windowing->input("frame");
  _windowing->output("frame")                 >>  _spectrum->input("frame");
  _spectrum->output("spectrum")               >>  _normSpectrum->input("array");
  _normSpectrum->output("array")              >>  _scaleSpectrum->input("array");
  _scaleSpectrum->output("array")             >>  _shiftSpectrum->input("array");
  _shiftSpectrum->output("array")             >>  _logSpectrum->input("array");
  _logSpectrum->output("array")               >>  _flux->input("spectrum");
  _flux->output("flux")                       >>  _lowPass->input("signal");
  // Compute Step 2 of algorithm
  _lowPass->output("signal")                  >>  _frameCutterOSS->input("signal");
  _frameCutterOSS->output("frame")            >>  _autoCorrelation->input("array");
  _autoCorrelation->output("autoCorrelation") >>  _enhanceHarmonics->input("array");
  _enhanceHarmonics->output("array")          >>  _peakDetection->input("array");
  _peakDetection->output("positions")         >>  _evaluatePulseTrains->input("positions");
  _peakDetection->output("amplitudes")        >>  NOWHERE;
  _frameCutterOSS->output("frame")            >>  _evaluatePulseTrains->input("oss");
  _evaluatePulseTrains->output("lag")         >>  PC(_pool, "lags");

  /*
  Algorithm* outSignalFrames = new FileOutput<std::vector<Real> >();
  outSignalFrames->configure("filename", "frames.txt", "mode", "text");
  _windowing->output("frame") >> outSignalFrames->input("data");

  Algorithm* outSpectrum = new FileOutput<std::vector<Real> >();
  outSpectrum->configure("filename", "spectrum.txt", "mode", "text");
  _logSpectrum->output("array") >> outSpectrum->input("data");

  Algorithm* outFlux = new FileOutput<Real >();
  outFlux->configure("filename", "flux.txt", "mode", "text");
  _flux->output("flux") >> outFlux->input("data");

  Algorithm* outLowpass = new FileOutput<Real >();
  outLowpass->configure("filename", "lowpass.txt", "mode", "text");
  _lowPass->output("signal") >> outLowpass->input("data");

  Algorithm* outOSSFrames = new FileOutput<std::vector<Real> >();
  outOSSFrames->configure("filename", "oss_frames.txt", "mode", "text");
  _frameCutterOSS->output("frame") >> outOSSFrames->input("data");

  Algorithm* outXcorr = new FileOutput<std::vector<Real> >();
  outXcorr->configure("filename", "xcorr.txt", "mode", "text");
  _autoCorrelation->output("autoCorrelation") >> outXcorr->input("data");

  Algorithm* outEXcorr = new FileOutput<std::vector<Real> >();
  outEXcorr->configure("filename", "excorr.txt", "mode", "text");
  _enhanceHarmonics->output("array") >> outEXcorr->input("data");
  */

  _network = new scheduler::Network(_frameCutter);
}

void TempoEstimator::clearAlgos() {
  if (!_configured) return;
  // it is safe to call this function here, as the inner network isn't connected to
  // anything outside, so it won't propagate and try to delete stuff twice
  delete _network;
}


TempoEstimator::~TempoEstimator() {
  clearAlgos();
}


void TempoEstimator::configure() {
  if (_configured) {
    clearAlgos();
  }

  _sampleRate   = parameter("sampleRate").toInt();
  _frameSize    = parameter("frameSize").toInt();
  _hopSize      = parameter("hopSize").toInt();
  _frameSizeOSS = parameter("frameSizeOSS").toInt();
  _hopSizeOSS   = parameter("hopSizeOSS").toInt();
  _minBPM       = parameter("minBPM").toInt();
  _maxBPM       = parameter("maxBPM").toInt();
  _srOSS        = (Real)_sampleRate / _hopSize;

  // TODO: Check that maxBPM > minBPM

  createInnerNetwork();

  // Configure internal algorithms
  _frameCutter->configure("frameSize", _frameSize,
                          "hopSize", _hopSize,
                          "startFromZero", true,
                          "validFrameThresholdRatio", 0,
                          "silentFrames", "keep");
  _windowing->configure("size", _frameSize,
                        "type", "hamming",
                        "normalized", false,
                        "zeroPhase", false);
  _spectrum->configure("size", _frameSize);
  _normSpectrum->configure("type", "identity", "scale", 1.0/_frameSize);
  _scaleSpectrum->configure("type", "identity", "scale", 1000.0);
  _shiftSpectrum->configure("type", "identity", "shift", 1.0);
  _logSpectrum->configure("type", "log");
  _flux->configure("halfRectify", true,
                   "norm", "L1");
  // NOTE: Original implementation uses 14 order low-pass FIR filter
  _lowPass->configure("cutoffFrequency", 7,
                      "sampleRate", 0.5 * (_sampleRate / _hopSize));
  _frameCutterOSS->configure("frameSize", _frameSizeOSS,
                             "hopSize", _hopSizeOSS,
                             "startFromZero", true,
                             "validFrameThresholdRatio", 0,
                             "silentFrames", "keep");
  _autoCorrelation->configure("normalization", "standard",
                              "generalized", true,
                              "frequencyDomainCompression", 0.5);
  _peakDetection->configure("maxPeaks", 10,
                            "range", _frameSizeOSS,
                            "minPosition", (int) (_srOSS * 60.0 / _maxBPM),
                            "maxPosition",  (int) (_srOSS * 60.0 / _minBPM),
                            "orderBy", "amplitude");
  _configured = true;

}

Real TempoEstimator::energyInRange(const std::vector<Real>& array,
                                   const Real low,
                                   const Real high,
                                   const Real scale) {
    int indexLow = round(low);
    int indexHigh = round(high);
    if (indexHigh > (array.size() - 1)){
      indexHigh = array.size() - 1;
    }
    if (indexLow < 0){
      indexLow = 0;
    }
    return scale * (Real)sum(array, indexLow, indexHigh + 1);
}

AlgorithmStatus TempoEstimator::process() {
  if (!shouldStop()) return PASS;

  // Compute Step 3 of algorithm

  // Create single gaussian template
  std::vector<Real> gaussian;
  int gaussianSize = 2000;
  gaussian.resize(gaussianSize);
  Real gaussianStd = 10.0;
  Real gaussianMean = gaussianSize / 2.0;
  for (int i=0; i < (int)(gaussianMean * 2); ++i){
    Real term1 = 1. / (gaussianStd * sqrt(2*M_PI));
    Real term2 = -2 * pow(gaussianStd, 2);
    gaussian[i] = term1 * exp(pow((i-gaussianMean), 2) / term2);
    //if (gaussian[i] < 1e-12) { // TODO: check if we need this
    //  gaussian[i] = 0; // null very low numbers to avoid numerical errors
    //}
  }

  // Accumulate (sum gaussians for every estimated lag)
  std::vector<Real> accum;
  accum.resize(414);  // 414 "long enough to accommodate all possible tempo lags"
  for (int i=0; i<_pool.value<vector<Real> >("lags").size(); ++i){
    int lag = (int)_pool.value<vector<Real> >("lags")[i];
    for (int j=0; j<accum.size(); ++j){
      accum[j] += gaussian[(int)(gaussianSize/2) - lag + j];
    }
  }

  // Pick highest peak
  int selectedLag = argmax(accum);
  Real bpm = _srOSS * 60.0 / selectedLag;

  // Octave decider (svm classifier)

  // Get features
  Real tolerance = 10.0;
  std::vector<Real> features;
  features.resize(3);
  Real energyTotal = energyInRange(accum, 0, accum.size() - 1, 1.0);
  Real energyUnder = energyInRange(accum, 0, selectedLag - tolerance, 1.0/energyTotal);
  Real str05 = energyInRange(accum, 0.5*selectedLag - tolerance, 0.5*selectedLag + tolerance, 1.0/energyTotal);
  features[0] = energyUnder;
  features[1] = str05;
  features[2] = bpm;

  // Hard-coded values provided by original authors
  // Apparently list initializer is not enabled so must initialize in this way...
  std::vector<Real> mins;
  mins.resize(3);
  mins[0] = 0.0321812;
  mins[1] = 1.68126e-83;
  mins[2] = 50.1745;
  std::vector<Real> maxs;
  maxs.resize(3);
  maxs[0] = 0.863237;
  maxs[1] = 0.449184;
  maxs[2] = 208.807;
  std::vector<Real> svmWeights51;
  svmWeights51.resize(4);
  svmWeights51[0] = -1.955100;
  svmWeights51[1] = 0.434800;
  svmWeights51[2] = -4.644200;
  svmWeights51[3] = 3.289600;
  std::vector<Real> svmWeights52;
  svmWeights52.resize(4);
  svmWeights52[0] = -3.040800;
  svmWeights52[1] = 2.759100;
  svmWeights52[2] = -6.536700;
  svmWeights52[3] = 3.081000;
  std::vector<Real> svmWeights12;
  svmWeights12.resize(4);
  svmWeights12[0] = -3.462400;
  svmWeights12[1] = 3.439700;
  svmWeights12[2] = -9.489700;
  svmWeights12[3] = 1.629700;

  // Normalize features
  for (int i=0; i<features.size(); ++i){
    features[i] = (features[i] - mins[i]) / (maxs[i] - mins[i]);
  }

  // Do classification to get the multiplier for the bpm
  Real svmSum51, svmSum52, svmSum12;
  svmSum51 = svmWeights51[svmWeights51.size()-1] + features[0] * svmWeights51[0] + features[1] * svmWeights51[1] + features[2] * svmWeights51[2];
  svmSum52 = svmWeights52[svmWeights52.size()-1] + features[0] * svmWeights52[0] + features[1] * svmWeights52[1] + features[2] * svmWeights52[2];
  svmSum12 = svmWeights12[svmWeights12.size()-1] + features[0] * svmWeights12[0] + features[1] * svmWeights12[1] + features[2] * svmWeights12[2];
  Real mult = 1.0;
  if ((svmSum52 > 0.0) && (svmSum12 > 0.0)){
    mult = 2.0;
  }
  if ((svmSum51 <= 0.0) && (svmSum52 <= 0.0)){
    mult = 0.5;
  }

  // Return final bpm
  _bpm.push(mult * bpm);
  return FINISHED;
}


void TempoEstimator::reset() {
  AlgorithmComposite::reset();
}

} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

const char* TempoEstimator::name = "TempoEstimator";
const char* TempoEstimator::description = DOC("This algorithm estimates the tempo in bpm from an input signal as described in [1]."
"Status: work in progress.\n"
"\n"
"\n"
"References:\n"
"  [1] Percival, G., & Tzanetakis, G. (2014). Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses.\n"
"  IEEE/ACM Transactions on Audio, Speech, and Language Processing, 22(12), 1765–1776.\n\n");

TempoEstimator::TempoEstimator() {
  declareInput(_signal, "signal", "input signal");
  declareOutput(_bpm, "bpm", "the tempo estimation [bpm]");
  createInnerNetwork();
}

TempoEstimator::~TempoEstimator() {
  delete _network;
}

void TempoEstimator::configure() {
  _tempoEstimator->configure(
    INHERIT("sampleRate"),
    INHERIT("frameSize"),
    INHERIT("hopSize"),
    INHERIT("frameSizeOSS"),
    INHERIT("hopSizeOSS"),
    INHERIT("minBPM"),
    INHERIT("maxBPM"));
}

void TempoEstimator::createInnerNetwork() {
  _tempoEstimator = streaming::AlgorithmFactory::create("TempoEstimator");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _tempoEstimator->input("signal");
  _tempoEstimator->output("bpm")  >>  PC(_pool, "bpm");

  _network = new scheduler::Network(_vectorInput);
}

void TempoEstimator::compute() {
  const vector<Real>& signal = _signal.get();
  Real& bpm = _bpm.get();

  _vectorInput->setVector(&signal);
  _network->run();
  bpm = _pool.value<Real >("bpm");
}

void TempoEstimator::reset() {
  _network->reset();
  _pool.remove("bpm");
}

} // namespace standard
} // namespace essentia
