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

#include "pitchyinprobabilistic.h"
#include <algorithm> // sort
#include "essentiamath.h"
#include <time.h>

using namespace std;

#include "poolstorage.h"

namespace essentia {
namespace streaming {

const char* PitchYinProbabilistic::name = essentia::standard::PitchYinProbabilistic::name;
const char* PitchYinProbabilistic::category = essentia::standard::PitchYinProbabilistic::category;
const char* PitchYinProbabilistic::description = essentia::standard::PitchYinProbabilistic::description;


PitchYinProbabilistic::PitchYinProbabilistic() : AlgorithmComposite() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _frameCutter       = factory.create("FrameCutter");
  _yinProbabilities  = factory.create("PitchYinProbabilities");
  _yinProbabilitiesHMM = standard::AlgorithmFactory::create("PitchYinProbabilitiesHMM");

  declareInput(_signal, "signal", "the input mono audio signal");
  declareOutput(_pitch, "pitch", "the output pitch estimations");
  declareOutput(_voicedProbabilities, "voicedProbabilities", "the voiced probabilities");
 
  // Connect input proxy
  _signal >> _frameCutter->input("signal");

  _frameCutter->output("frame") >> _yinProbabilities->input("signal");
  _yinProbabilities->output("pitch") >> PC(_pool, "frequencies");
  _yinProbabilities->output("probabilities") >> PC(_pool, "probabilities");
  _yinProbabilities->output("RMS") >> PC(_pool, "RMS");

  _network = new scheduler::Network(_frameCutter);
}

PitchYinProbabilistic::~PitchYinProbabilistic() {
  delete _network;
}

void PitchYinProbabilistic::configure() {

  Real sampleRate = parameter("sampleRate").toReal();
  _frameSize = parameter("frameSize").toInt();
  _hopSize = parameter("hopSize").toInt();
  _lowRMSThreshold = parameter("lowRMSThreshold").toReal();
  _outputUnvoiced = parameter("outputUnvoiced").toString();
  _preciseTime = parameter("preciseTime").toBool();
  
  _frameCutter->configure("frameSize", _frameSize,
                          "hopSize", _hopSize,
                          "startFromZero", true,
                          "silentFrames", "keep");

  _yinProbabilities->configure("frameSize", _frameSize,
                               "sampleRate", sampleRate,
                               "lowAmp", _lowRMSThreshold,
                               "preciseTime", _preciseTime);
}


AlgorithmStatus PitchYinProbabilistic::process() {
  if (!shouldStop()) return PASS;

  if (!_pool.contains<vector<vector<Real> > >("frequencies") || !_pool.contains<vector<vector<Real> > >("probabilities") || !_pool.contains<vector<Real> >("RMS")) {
    // do not push anything in the case of empty signal
    E_WARNING("PitchYinProbabilistic: empty input signal");
    return FINISHED;
  }

  const vector<vector<Real> >& pitchCandidates = _pool.value<vector<vector<Real> > >("frequencies");
  const vector<vector<Real> >& probabilities = _pool.value<vector<vector<Real> > >("probabilities");

  vector<Real> tempPitch;
  _yinProbabilitiesHMM->input("pitchCandidates").set(pitchCandidates);
  _yinProbabilitiesHMM->input("probabilities").set(probabilities);
  _yinProbabilitiesHMM->output("pitch").set(tempPitch);
  _yinProbabilitiesHMM->compute();

  // voiced probabilities
  vector<Real> oF0Probs (probabilities.size(), 0.0);
  for (size_t j = 0; j < probabilities.size(); ++j) {  
    Real voicedProb = 0;
    for (size_t i = 0; i < probabilities[j].size(); ++i) {
        voicedProb += probabilities[j][i];
    }
    oF0Probs[j] = voicedProb;
  }
  _voicedProbabilities.push(oF0Probs);


  vector<Real> _tempPitchVoicing(tempPitch.size());
  for (size_t iFrame = 0; iFrame < tempPitch.size(); ++iFrame) {
    if (tempPitch[iFrame] < 0 && (_outputUnvoiced=="zero")) continue;
    if (_outputUnvoiced == "abs") {
      _tempPitchVoicing[iFrame] = fabs(tempPitch[iFrame]);
    } else {
      _tempPitchVoicing[iFrame] = tempPitch[iFrame];
    }    
  }

  _pitch.push(_tempPitchVoicing);

  return FINISHED;
}

void PitchYinProbabilistic::reset() {
  AlgorithmComposite::reset();
  _pool.remove("frequencies");
  _pool.remove("probabilities");
  _pool.remove("RMS");
}

} // namespace streaming
} // namespace essentia


namespace essentia {
namespace standard {

const char* PitchYinProbabilistic::name = "PitchYinProbabilistic";
const char* PitchYinProbabilistic::category = "Pitch";
const char* PitchYinProbabilistic::description = DOC("This algorithm computes the pitch track of a mono audio signal using probabilistic Yin algorithm.\n"
"\n"
"- The input mono audio signal is preprocessed with a FrameCutter to segment into frameSize chunks with a overlap hopSize.\n"
"- The pitch frequencies, probabilities and RMS values of the chunks are then calculated by PitchYinProbabilities algorithm. The results of all chunks are aggregated into a Essentia pool.\n"
"- The pitch frequencies and probabilities are finally sent to PitchYinProbabilitiesHMM algorithm to get a smoothed pitch track and a voiced probability.\n"
"\n"
"References:\n"
"  [1] M. Mauch and S. Dixon, \"pYIN: A Fundamental Frequency Estimator\n"
"  Using Probabilistic Threshold Distributions,\" in Proceedings of the\n"
"  IEEE International Conference on Acoustics, Speech, and Signal Processing\n"
"  (ICASSP 2014)Project Report, 2004");


PitchYinProbabilistic::PitchYinProbabilistic() {
  declareInput(_signal, "signal", "the input mono audio signal");
  declareOutput(_pitch, "pitch", "the output pitch estimations");
  declareOutput(_voicedProbabilities, "voicedProbabilities", "the voiced probabilities");

  createInnerNetwork();
}

PitchYinProbabilistic::~PitchYinProbabilistic() {
  delete _network;
}

void PitchYinProbabilistic::configure() {

  _PitchYinProbabilistic->configure(INHERIT("sampleRate"), 
                                    INHERIT("frameSize"),
                                    INHERIT("hopSize"),
                                    INHERIT("lowRMSThreshold"),
                                    INHERIT("outputUnvoiced"),
                                    INHERIT("preciseTime"));
}


void PitchYinProbabilistic::createInnerNetwork() {
  _PitchYinProbabilistic = streaming::AlgorithmFactory::create("PitchYinProbabilistic");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _PitchYinProbabilistic->input("signal");
  _PitchYinProbabilistic->output("pitch") >> PC(_pool, "pitch");
  _PitchYinProbabilistic->output("voicedProbabilities") >> PC(_pool, "voicedProbabilities");
  
  _network = new scheduler::Network(_vectorInput);
}

void PitchYinProbabilistic::compute() {
  const vector<Real>& signal = _signal.get();
  if (!signal.size()) {
    throw EssentiaException("PitchYinProbabilistic: empty input signal");
  }

  _vectorInput->setVector(&signal);
  _network->run();

  vector<Real>& pitch = _pitch.get();
  vector<Real>& voicedProbas = _voicedProbabilities.get();

  pitch = _pool.value<vector<Real> >("pitch");
  voicedProbas = _pool.value<vector<Real> >("voicedProbabilities");

  reset();
}

void PitchYinProbabilistic::reset() {
  _network->reset();
  _pool.remove("pitch");  
  _pool.remove("voicedProbs");
  _pool.remove("RMS");
}

} // namespace standard
} // namespace essentia
