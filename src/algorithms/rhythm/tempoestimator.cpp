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
  _frameCutterOSS(0), _autoCorrelation(0), _peakDetection(0), _configured(false) {
  declareInput(_signal, "signal", "input signal");
  declareOutput(_bpm, 0, "bpm", "the tempo estimation [bpm]");
}

void TempoEstimator::createInnerNetwork() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter      = factory.create("FrameCutter");
  _windowing        = factory.create("Windowing");
  _spectrum         = factory.create("Spectrum");
  _scaleSpectrum    = factory.create("UnaryOperator");
  _shiftSpectrum    = factory.create("UnaryOperator");
  _logSpectrum    = factory.create("UnaryOperator");
  _normSpectrum    = factory.create("UnaryOperator");
  _flux             = factory.create("Flux");
  _lowPass          = factory.create("LowPass");
  _frameCutterOSS   = factory.create("FrameCutter");
  _autoCorrelation  = factory.create("AutoCorrelation");
  _peakDetection    = factory.create("PeakDetection");

  // Connect internal algorithms
  _signal                                     >>  _frameCutter->input("signal");
  _frameCutter->output("frame")               >>  _windowing->input("frame");
  _windowing->output("frame")                 >>  _spectrum->input("frame");
  _spectrum->output("spectrum")               >>  _normSpectrum->input("array");
  _normSpectrum->output("array")              >>  _scaleSpectrum->input("array");
  _scaleSpectrum->output("array")             >>  _shiftSpectrum->input("array");
  _shiftSpectrum->output("array")             >>  _logSpectrum->input("array");
  _logSpectrum->output("array")               >>  _flux->input("spectrum");
  _flux->output("flux")                       >>  _lowPass->input("signal");
  _lowPass->output("signal")                  >>  _frameCutterOSS->input("signal");
  _frameCutterOSS->output("frame")            >>  _autoCorrelation->input("array");
  _autoCorrelation->output("autoCorrelation") >>  _peakDetection->input("array");
  _peakDetection->output("positions")         >>  NOWHERE;


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

  Algorithm* outPeaks = new FileOutput<std::vector<Real> >();
  outPeaks->configure("filename", "peaks.txt", "mode", "text");
  _peakDetection->output("positions") >> outPeaks->input("data");
  

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

  createInnerNetwork();


  // Configure internal algorithms
  _frameCutter->configure("frameSize", _frameSize,
                          "hopSize", _hopSize,
                          "startFromZero", true,
                          "validFrameThresholdRatio", 1,
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
  _lowPass->configure("cutoffFrequency", 7,
                      "sampleRate", 0.5 * (_sampleRate / _hopSize));
  _frameCutterOSS->configure("frameSize", _frameSizeOSS,
                             "hopSize", _hopSizeOSS,
                             "startFromZero", true,
                             "validFrameThresholdRatio", 1,
                             "silentFrames", "keep");
  _autoCorrelation->configure("normalization", "standard",
                              "generalized", true,
                              "frequencyDomainCompression", 0.5);
  _peakDetection->configure("maxPeaks", 10,
                            "range", 1000.0,
                            "minPosition", 98, // TODO: Should depend on min/max bpm parameters
                            "maxPosition",  414 );
  _configured = true;

}

AlgorithmStatus TempoEstimator::process() {
  if (!shouldStop()) return PASS;

  Real bpm = 120.0;  
  _bpm.push(bpm);
  return FINISHED;
}


void TempoEstimator::reset() {
  AlgorithmComposite::reset();
}

} // namespace streaming
} // namespace essentia
