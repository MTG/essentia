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
#include "tnt/tnt.h"
#include "essentiamath.h"
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
  : AlgorithmComposite(), _frameCutter(0), _powerSpectrum(0), _flux(0), _lowPass(0), 
  _fileOutputOSS(0), _frameCutterOSS(0), _autoCorrelation(0), _peakDetection(0), _configured(false) {
  declareInput(_signal, "signal", "input signal");
  declareOutput(_bpm, 0, "bpm", "the tempo estimation [bpm]");
}

void TempoEstimator::createInnerNetwork() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter      = factory.create("FrameCutter");
  _powerSpectrum    = factory.create("PowerSpectrum");
  _flux             = factory.create("Flux");
  _lowPass          = factory.create("LowPass");
  _frameCutterOSS   = factory.create("FrameCutter");
  _autoCorrelation  = factory.create("AutoCorrelation");
  _peakDetection    = factory.create("PeakDetection");

  _fileOutputOSS    = new FileOutput<Real >();

  // Connect internal algorithms
  _signal                                     >>  _frameCutter->input("signal");
  _frameCutter->output("frame")               >>  _powerSpectrum->input("signal");
  _powerSpectrum->output("powerSpectrum")     >>  _flux->input("spectrum");
  _flux->output("flux")                       >>  _lowPass->input("signal");
  _lowPass->output("signal")                  >>  _frameCutterOSS->input("signal");
  _frameCutterOSS->output("frame")            >>  _autoCorrelation->input("array");
  _autoCorrelation->output("autoCorrelation") >>  _peakDetection->input("array");
  _peakDetection->output("positions")         >>  NOWHERE;

  _flux->output("flux")                       >>  _fileOutputOSS->input("data");
  
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
                          "hopSize", _hopSize);
  _flux->configure("halfRectify", false,
                   "norm", "L2");
  _lowPass->configure("cutoffFrequency", 7,
                      "sampleRate", _sampleRate);
  _frameCutterOSS->configure("frameSize", _frameSizeOSS,
                             "hopSize", _hopSizeOSS);
  _peakDetection->configure("maxPeaks", 10);
  _fileOutputOSS->configure("filename", "oss.txt", "mode", "text");
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
