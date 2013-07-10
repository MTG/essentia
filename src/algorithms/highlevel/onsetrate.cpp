/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

#include "onsetrate.h"
#include <complex>
#include "tnt/tnt.h"

using namespace std;
using namespace TNT;

namespace essentia {
namespace standard {


const char* OnsetRate::name = "OnsetRate";
const char* OnsetRate::description = DOC("Given an audio signal, this algorithm outputs the rate at which onsets occur and the onsets' position in time. Onset detection functions are computed using both high frequency content and complex-domain methods available in OnsetDetection algorithm. See OnsetDetection for more information.\n"
"Please note that due to a dependence on the Onsets algorithm, this algorithm is only valid for audio signals with a sampling rate of 44100Hz.\n"
"This algorithm throws an exception if the input signal is empty.");

void OnsetRate::configure() {
  _sampleRate = 44100.0;
  _frameSize = 1024;
  _hopSize = 512;
  _frameRate = _sampleRate/Real(_hopSize);
  _zeroPadding = 0;

  // Pre-processing
  _frameCutter->configure("frameSize", _frameSize,
                           "hopSize", _hopSize,
                           "startFromZero", true);

  _windowing->configure("size", _frameSize,
                        "zeroPadding", _zeroPadding,
                        "type", "hann");
  // FFT
  _fft->configure("size", _frameSize + _zeroPadding);

  // Onsets
  _onsetHfc->configure("method", "hfc",
                  "sampleRate", _sampleRate);

  _onsetComplex->configure("method", "complex",
                      "sampleRate", _sampleRate);

  _onsets->configure("frameRate", _frameRate);
}

void OnsetRate::compute() {
  const vector<Real>& signal = _signal.get();
  Real& onsetRate = _onsetRate.get();
  vector<Real>& onsetTimes = _onsetTimes.get();
  if (signal.empty()) {
    throw EssentiaException("OnsetRate: empty input signal");
  }

  // Pre-processing
  vector<Real> frame;
  _frameCutter->input("signal").set(signal);
  _frameCutter->output("frame").set(frame);

  vector<Real> frameWindowed;
  _windowing->input("frame").set(frame);
  _windowing->output("frame").set(frameWindowed);

  vector<complex<Real> > frameFFT;
  _fft->input("frame").set(frameWindowed);
  _fft->output("fft").set(frameFFT);

  vector<Real> frameSpectrum;
  vector<Real> framePhase;
  _cartesian2polar->input("complex").set(frameFFT);
  _cartesian2polar->output("magnitude").set(frameSpectrum);
  _cartesian2polar->output("phase").set(framePhase);

  Real frameHFC;
  _onsetHfc->input("spectrum").set(frameSpectrum);
  _onsetHfc->input("phase").set(framePhase);
  _onsetHfc->output("onsetDetection").set(frameHFC);

  Real frameComplex;
  _onsetComplex->input("spectrum").set(frameSpectrum);
  _onsetComplex->input("phase").set(framePhase);
  _onsetComplex->output("onsetDetection").set(frameComplex);

  vector<Real> hfc;
  vector<Real> complexdomain;

  while (true) {
    // get a frame
    _frameCutter->compute();


    if (!frame.size()) {
      break;
    }

    _windowing->compute();

    // calculate fft
    _fft->compute();

    // calculate magnitude/phase
    _cartesian2polar->compute();

    // calculate hfc onset
    _onsetHfc->compute();

    // calculate complex onset
    _onsetComplex->compute();

    hfc.push_back(frameHFC);
    complexdomain.push_back(frameComplex);
  }

  // Time onsets
  Array2D<Real> detections;
  detections = Array2D<Real>(2, hfc.size());

  for (int j=0; j<int(hfc.size()); ++j) {
    detections[0][j] = hfc[j];
    detections[1][j] = complexdomain[j];
  }

  vector<Real> weights(2);
  weights[0] = 1.0;
  weights[1] = 1.0;

  _onsets->input("detections").set(detections);
  _onsets->input("weights").set(weights);
  _onsets->output("onsets").set(onsetTimes);
  _onsets->compute();

  onsetRate = onsetTimes.size() / (signal.size() / _sampleRate);
}

OnsetRate::~OnsetRate() {
    // Pre-processing
    delete _frameCutter;
    delete _windowing;

    // FFT
    delete _fft;
    delete _cartesian2polar;

    // Onsets
    delete _onsetHfc;
    delete _onsetComplex;
    delete _onsets;
}


} // namespace standard
} // namespace essentia

#include "poolstorage.h"

namespace essentia {
namespace streaming {


OnsetRate::OnsetRate() : AlgorithmComposite() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _frameCutter  = factory.create("FrameCutter");
  _windowing    = factory.create("Windowing");
  _fft          = factory.create("FFT");
  _cart2polar   = factory.create("CartesianToPolar");
  _onsetHfc     = factory.create("OnsetDetection");
  _onsetComplex = factory.create("OnsetDetection");

  _onsets = standard::AlgorithmFactory::create("Onsets");

  _preferredBufferSize = 1024;

  declareInput(_signal, _preferredBufferSize, "signal", "the input audio signal");

  // Connect internal algorithms
  declareOutput(_onsetTimes, 0, "onsetTimes", "the detected onset times [s]");
  declareOutput(_onsetRate, 0, "onsetRate", "the number of onsets per second");

  _signal  >> _frameCutter->input("signal");

  _frameCutter->output("frame")     >>  _windowing->input("frame");
  _windowing->output("frame")       >>  _fft->input("frame");
  _fft->output("fft")               >>  _cart2polar->input("complex");

  _cart2polar->output("magnitude")  >>  _onsetHfc->input("spectrum");
  _cart2polar->output("phase")      >>  _onsetHfc->input("phase");

  _cart2polar->output("magnitude")  >>  _onsetComplex->input("spectrum");
  _cart2polar->output("phase")      >>  _onsetComplex->input("phase");

  _onsetHfc->output("onsetDetection")      >>  PC(_pool, "internal.hfc");
  _onsetComplex->output("onsetDetection")  >>  PC(_pool, "internal.complexdomain");

  _network = new scheduler::Network(_frameCutter);
}

OnsetRate::~OnsetRate() {
  delete _network;
  delete _onsets;
}

void OnsetRate::configure() {
  _sampleRate = 44100.0;
  _frameSize = 1024;
  _hopSize = 512;
  _frameRate = (Real)_sampleRate/_hopSize;
  _zeroPadding = 0;

  // Pre-processing
  _frameCutter->configure("frameSize", _frameSize,
                          "hopSize", _hopSize,
                          "silentFrames", "keep");
  // don't add noise, as completely empty signals will yield 1 onset at the
  // begining

  _windowing->configure("size", _frameSize,
                        "zeroPadding", _zeroPadding,
                        "type", "hann");

  // FFT
  _fft->configure("size", _frameSize + _zeroPadding);

  // Onsets
  _onsetHfc->configure("method", "hfc",
                       "sampleRate", _sampleRate);

  _onsetComplex->configure("method", "complex",
                           "sampleRate", _sampleRate);

  _onsets->configure("frameRate", _frameRate);
}

AlgorithmStatus OnsetRate::process() {
  if (!shouldStop()) return PASS;

  const vector<Real>& hfc = _pool.value<vector<Real> >("internal.hfc");
  const vector<Real>& complexdomain = _pool.value<vector<Real> >("internal.complexdomain");
  // Time onsets
  TNT::Array2D<Real> detections;
  vector<Real> onsetTimes;
  detections = TNT::Array2D<Real>(2, hfc.size());

  for (int j=0; j<int(hfc.size()); j++) {
    detections[0][j] = hfc[j];
    detections[1][j] = complexdomain[j];
  }

  vector<Real> weights(2);
  weights[0] = 1.0;
  weights[1] = 1.0;

  _onsets->input("detections").set(detections);
  _onsets->input("weights").set(weights);
  _onsets->output("onsets").set(onsetTimes);
  _onsets->compute();

  _onsetTimes.push(onsetTimes);
  // the size of the signal is taken as the size of the dectection function
  // multiplied by the hopsize. This is not 100% accurate but it approximates
  // ok
  _onsetRate.push(Real(onsetTimes.size()) / Real(hfc.size()*_hopSize) / _sampleRate);

  return FINISHED;
}

void OnsetRate::reset() {
  AlgorithmComposite::reset();
  _onsets->reset();
}


} // namespace streaming
} // namespace essentia
