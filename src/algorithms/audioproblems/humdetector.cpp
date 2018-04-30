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

#include "humdetector.h"
#include <algorithm> // sort
#include "essentiamath.h"

using namespace std;

#include "poolstorage.h"

namespace essentia {
namespace streaming {

const char* HumDetector::name = essentia::standard::HumDetector::name;
const char* HumDetector::category = essentia::standard::HumDetector::category;
const char* HumDetector::description = essentia::standard::HumDetector::description;


template< typename T >
typename std::vector<T>::iterator 
   HumDetector::insertSorted( std::vector<T> & vec, T const& item )
{
    return vec.insert
        ( 
            std::upper_bound( vec.begin(), vec.end(), item ),
            item 
        );
}

HumDetector::HumDetector() : AlgorithmComposite() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _decimator                = factory.create("Resample");
  _lowPass                  = factory.create("LowPass");
  _frameCutter              = factory.create("FrameCutter");
  _welch                    = factory.create("PowerSpectrum");

  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_frequencies, "frequencies", "humming tones frequencies");
  declareOutput(_amplitudes, "amplitudes", "humming tones amplitudes");
  declareOutput(_starts, "starts", "humming tones starts");
  declareOutput(_ends, "ends", "humming tones ends");

  // Connect input proxy
  _signal >> _decimator->input("signal");

  _decimator->output("signal").setBufferType(BufferUsage::forLargeAudioStream);
  
  _decimator->output("signal") >> _lowPass->input("signal");

  _lowPass->output("signal").setBufferType(BufferUsage::forLargeAudioStream);

  _lowPass->output("signal") >> _frameCutter->input("signal");

  _frameCutter->output("frame") >> _welch->input("signal");


  _welch->output("powerSpectrum") >> PC(_pool, "psd");

  _network = new scheduler::Network(_decimator);
}

HumDetector::~HumDetector() {
  delete _network;
}


void HumDetector::configure() {

  _outSampleRate = 1000.f;
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = int(round(parameter("hopSize").toReal() * _outSampleRate));
  _frameSize = int(round(parameter("frameSize").toReal() * _outSampleRate));
  _timeWindow = int(round(parameter("timeWindow").toReal() * _outSampleRate / _hopSize));
  _Q0 = parameter("Q0").toReal();
  _Q1 = parameter("Q1").toReal();

  _Q0sample = (uint)(_Q0 * _timeWindow + 0.5);
  _Q1sample = (uint)(_Q1 * _timeWindow + 0.5);


  _decimator->configure("inputSampleRate", _sampleRate,
                        "outputSampleRate", _outSampleRate);
  
  _lowPass->configure("sampleRate",_outSampleRate,
                      "cutoffFrequency", 900.f);

  _frameCutter->configure("frameSize",_frameSize,
                          "hopSize", _hopSize,
                          "silentFrames", "keep");

  _welch->configure("size",_frameSize);

}


AlgorithmStatus HumDetector::process() {
  if (!shouldStop()) return PASS;

  if (!_pool.contains<vector<vector<Real> > >("psd")) {
    // do not push anything in the case of empty signal
    E_WARNING("HumDetector: empty input signal");
    return FINISHED;
  }

  const vector<vector<Real> >& psd = _pool.value<vector<vector<Real> > >("psd");

  _spectSize = psd[0].size();
  _timeStamps = psd.size();
  std::vector<vector<Real> >psdWindow(_spectSize, vector<Real>(_timeWindow, 0.f));
  std::vector<vector<size_t> >psdIdxs(_spectSize, vector<size_t>(_timeWindow, 0));


  std::vector<vector<Real> >r(_spectSize, vector<Real>());

  Real R0, R1;
  for (uint i = 0; i < _spectSize; i++) {
    for (uint j = 0; j < _timeWindow; j++)
      psdWindow[i][j] = psd[j][i];

    sort(psdWindow[i].begin(), psdWindow[i].end());
    Real R0 = psdWindow[i][_Q0sample];
    Real R1 = psdWindow[i][_Q1sample];
    
    r[i].push_back(R1 - R0);
  }

  for (uint i = 0; i < _spectSize; i++) {
    for (uint j = _timeWindow; j < _timeStamps; j++) {
      rotate(psdWindow.begin(), psdWindow.begin() + 1, psdWindow.end());
      insertSorted(psdWindow[i], psd[j][i]);
      psdWindow[i].pop_back();

      Real R0 = psdWindow[i][_Q0sample];
      Real R1 = psdWindow[i][_Q1sample];

      r[i].push_back(R1 - R0);
      }
  }

}

void HumDetector::reset() {
  AlgorithmComposite::reset();
  _pool.remove("psd");
}

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

const char* HumDetector::name = "HumDetector";
const char* HumDetector::category = "Audio Problems";
const char* HumDetector::description = DOC("");


HumDetector::HumDetector() {
  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_frequencies, "frequencies", "humming tones frequencies");
  declareOutput(_amplitudes, "amplitudes", "humming tones amplitudes");
  declareOutput(_starts, "starts", "humming tones starts");
  declareOutput(_ends, "ends", "humming tones ends");

  createInnerNetwork();
}

HumDetector::~HumDetector() {
  delete _network;
}

void HumDetector::configure() {
  _humDetector->configure(INHERIT("sampleRate"), INHERIT("hopSize"),
                          INHERIT("frameSize"), INHERIT("timeWindow"));
}


void HumDetector::createInnerNetwork() {
  _humDetector = streaming::AlgorithmFactory::create("HumDetector");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _humDetector->input("signal");
  _humDetector->output("frequencies")    >> PC(_pool, "frequencies");
  _humDetector->output("amplitudes")    >> PC(_pool, "amplitudes");
  _humDetector->output("starts")   >> PC(_pool, "starts");
  _humDetector->output("ends")        >> PC(_pool, "ends");

  _network = new scheduler::Network(_vectorInput);
}

void HumDetector::compute() {
  const vector<Real>& signal = _signal.get();
  if (!signal.size()) {
    throw EssentiaException("HumDetector: empty input signal");
  }

  _vectorInput->setVector(&signal);
  _network->run();

  vector<Real>& frequencies = _frequencies.get();
  vector<Real>& amplitudes = _amplitudes.get();
  vector<Real>& starts = _starts.get();
  vector<Real>& ends = _ends.get();


  frequencies = _pool.value<vector<Real> >("frequencies");
  amplitudes = _pool.value<vector<Real> >("amplitudes");
  starts = _pool.value<vector<Real> >("starts");
  ends = _pool.value<vector<Real> >("ends");

  reset();
}

void HumDetector::reset() {
  _network->reset();
  _pool.remove("momentaryLoudness");
  _pool.remove("shortTermLoudness");
  _pool.remove("integratedLoudness");
  _pool.remove("loudnessRange");
  //_pool.remove("momentaryLoudnessMax");
  //_pool.remove("shortTermLoudnessMax");
}

} // namespace standard
} // namespace essentia
