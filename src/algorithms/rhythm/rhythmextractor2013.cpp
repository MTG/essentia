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

#include <complex>
#include "rhythmextractor2013.h"
#include "tnt/tnt.h"
#include "essentiamath.h"
#include "poolstorage.h"
#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* RhythmExtractor2013::name = "RhythmExtractor2013";
const char* RhythmExtractor2013::description = DOC("This algorithm estimates the beat locations and the confidence of their estimation given an input signal, as well as its tempo in bpm. The beat locations can be computed using:\n"
"  - 'multifeature', the BeatTrackerMultiFeature algorithm\n"
"  - 'degara', the BeatTrackerDegara algorithm (note that there is no confidence estimation for this method, the output confidence value is always 0)\n"
"\n"
"See BeatTrackerMultiFeature and  BeatTrackerDegara algorithms for more details.\n"
"\n"
"Note that the algorithm requires the sample rate of the input signal to be 44100 Hz in order to work correctly.\n");


RhythmExtractor2013::RhythmExtractor2013() : AlgorithmComposite() {
  _configured = false;

  declareInput(_signal, "signal", "input signal");

  declareOutput(_ticks, "ticks", " the estimated tick locations [s]");
  declareOutput(_confidence, "confidence", "confidence with which the ticks are detected (ignore this value if using 'degara' method)");
  declareOutput(_bpm, 0, "bpm", "the tempo estimation [bpm]");
  declareOutput(_estimates, 0, "estimates", "the list of bpm estimates characterizing the bpm distribution for the signal [bpm]");
  //TODO we need better rubato estimation algorithm
  //declareOutput(_rubatoStart, 0, "rubatoStart", "list of start times for rubato section [s]");
  //declareOutput(_rubatoStop, 0, "rubatoStop", "list of stop times of rubato section [s]");
  //declareOutput(_rubatoNumber, 0, "rubatoNumber", "number of rubato sections");
  declareOutput(_bpmIntervals, 0, "bpmIntervals", "list of beats interval [s]");
}

void RhythmExtractor2013::createInnerNetwork() {

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _method = parameter("method").toLower();
  if (_method == "multifeature") {
    _beatTracker = factory.create("BeatTrackerMultiFeature");
    _beatTracker->output("confidence") >> PC(_pool, "internal.confidence");
  }
  else if (_method == "degara") {
    _beatTracker = factory.create("BeatTrackerDegara");
  }
  //_bpmRubato = standard::AlgorithmFactory::create("BpmRubato");

  // Connect internal algorithms
  _signal                           >>  _beatTracker->input("signal");
  //_beatTracker->output("ticks")     >>  _ticks;
  _beatTracker->output("ticks")     >>  PC(_pool, "internal.ticks");


  // TODO change SinkProxy in BpmRubato to Real?
  //_beatTracker->output("ticks")     >>  _bpmRubato->input("beats");
  //_bpmRubato->output("rubatoStart") >>  _rubatoStart;
  //_bpmRubato->output("rubatoStop")  >>  _rubatoStop;

  _network = new scheduler::Network(_beatTracker);
}


RhythmExtractor2013::~RhythmExtractor2013() {
  clearAlgos();
}

void RhythmExtractor2013::clearAlgos() {
  if (!_configured) return;
  delete _network;
  //delete _bpmRubato;
}

void RhythmExtractor2013::configure() {
   if (_configured) {
    clearAlgos();
  }

  _periodTolerance = 5.;

  createInnerNetwork();

  // Configure internal algorithms
  _beatTracker->configure(INHERIT("minTempo"), INHERIT("maxTempo"));
  _configured = true;

}

AlgorithmStatus RhythmExtractor2013::process() {
  if (!shouldStop()) return PASS;

  // 'degara' method does not output confidence
  if (_method == "multifeature") {
    _confidence.push(_pool.value<Real>("internal.confidence"));
  }
  else if (_method == "degara") {
    _confidence.push((Real) 0);
  }
  
  vector<Real> bpmIntervals;
  vector<Real> bpmEstimateList;

  // push ticks if any, otherwise push an empty vector 
  if (_pool.contains<vector<Real> >("internal.ticks")) {
    const vector<Real>& ticks = _pool.value<vector<Real> >("internal.ticks");
    _ticks.push(ticks);

    if (ticks.size() > 1) {
      // computing beats intervals
      bpmIntervals.reserve(ticks.size() - 1);
      bpmEstimateList.reserve(bpmIntervals.size());
      for (size_t i = 1; i < ticks.size(); i++) {
        bpmIntervals.push_back(ticks[i] - ticks[i-1]);
        bpmEstimateList.push_back(60. / bpmIntervals.back()); // period to bpm
      }

      // computing rubato regions
      //vector<Real> rubatoStart, rubatoStop;
      //int rubatoNumber;
      //_bpmRubato->input("beats").set(ticks);
      //_bpmRubato->output("rubatoStart").set(rubatoStart);
      //_bpmRubato->output("rubatoStop").set(rubatoStop);
      //_bpmRubato->output("rubatoNumber").set(rubatoNumber);
      //_bpmRubato->compute();

      //_rubatoStart.push(rubatoStart);
      //_rubatoStop.push(rubatoStop);
      //_rubatoNumber.push(rubatoNumber);
    }
  }
  else {
    _ticks.push(vector<Real>());
  }

  _bpmIntervals.push(bpmIntervals);

  // estimate bpm. TODO why is _periodTolerance necessary? MAGIC NUMBER?
  vector<Real> estimates;
  Real bpm;
  if (bpmEstimateList.size() > 0) {
    Real closestBpm = 0;
    vector<Real> countedBins;
    for (size_t i=0; i < bpmEstimateList.size(); ++i) {
      bpmEstimateList[i] /= 2.;
    }
    bincount(bpmEstimateList, countedBins);
    closestBpm = argmax(countedBins) * 2;
    for (size_t i=0; i < bpmEstimateList.size(); ++i) {
      bpmEstimateList[i] *= 2.;
      if (abs(closestBpm - bpmEstimateList[i]) < _periodTolerance) {
        estimates.push_back(bpmEstimateList[i]);
      }
    }
    if (estimates.size() < 1) {
      // something odd happened
      bpm = closestBpm;
    }
    else {
      bpm = mean(estimates);
    }
  }
  else {
    bpm = 0.;
  }
  _bpm.push(bpm);
  _estimates.push(estimates);

  return FINISHED;
}


void RhythmExtractor2013::reset() {
  AlgorithmComposite::reset();
}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* RhythmExtractor2013::name = "RhythmExtractor2013";
const char* RhythmExtractor2013::description = DOC("This algorithm estimates the beat locations and the confidence of their estimation given an input signal, as well as its tempo in bpm. The beat locations can be computed using:\n"
"  - 'multifeature', the BeatTrackerMultiFeature algorithm\n"
"  - 'degara', the BeatTrackerDegara algorithm (note that there is no confidence estimation for this method, the output confidence value is always 0)\n"
"\n"
"See BeatTrackerMultiFeature and  BeatTrackerDegara algorithms for more details.\n"
"\n"
"Note that the algorithm requires the sample rate of the input signal to be 44100 Hz in order to work correctly.\n");


RhythmExtractor2013::RhythmExtractor2013() {
  declareInput(_signal, "signal", "the audio input signal");
  declareOutput(_bpm, "bpm", "the tempo estimation [bpm]");
  declareOutput(_ticks, "ticks", " the estimated tick locations [s]");
  declareOutput(_confidence, "confidence", "confidence with which the ticks are detected (ignore this value if using 'degara' method)");
  declareOutput(_estimates, "estimates", "the list of bpm estimates characterizing the bpm distribution for the signal [bpm]");
  //declareOutput(_rubatoStart, "rubatoStart", "list of start times for rubato section [s]");
  //declareOutput(_rubatoStop, "rubatoStop", "list of stop times of rubato section [s]");
  //declareOutput(_rubatoNumber, "rubatoNumber", "number of rubato sections");
  declareOutput(_bpmIntervals, "bpmIntervals", "list of beats interval [s]");

  createInnerNetwork();
}

RhythmExtractor2013::~RhythmExtractor2013() {
  delete _network;
}

void RhythmExtractor2013::configure() {
  _rhythmExtractor->configure(INHERIT("maxTempo"), INHERIT("minTempo"),
                              INHERIT("method"));
}


void RhythmExtractor2013::createInnerNetwork() {
  _rhythmExtractor = streaming::AlgorithmFactory::create("RhythmExtractor2013");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _rhythmExtractor->input("signal");
  _rhythmExtractor->output("ticks")         >>  PC(_pool, "internal.ticks");
  _rhythmExtractor->output("confidence")    >>  PC(_pool, "internal.confidence");
  _rhythmExtractor->output("bpm")           >>  PC(_pool, "internal.bpm");
  _rhythmExtractor->output("estimates")     >>  PC(_pool, "internal.estimates");
  //_rhythmExtractor->output("rubatoStart")   >>  PC(_pool, "internal.rubatoStart");
  //_rhythmExtractor->output("rubatoStop")    >>  PC(_pool, "internal.rubatoStop");
  //_rhythmExtractor->output("rubatoNumber")  >>  PC(_pool, "internal.rubatoNumber");
  _rhythmExtractor->output("bpmIntervals")  >>  PC(_pool, "internal.bpmIntervals");

  _network = new scheduler::Network(_vectorInput);
}

void RhythmExtractor2013::compute() {
  const vector<Real>& signal = _signal.get();
  _vectorInput->setVector(&signal);

  _network->run();

  Real& bpm = _bpm.get();
  vector<Real>& ticks = _ticks.get();
  Real& confidence = _confidence.get();
  vector<Real>& estimates = _estimates.get();
  //vector<Real>& rubatoStart = _rubatoStart.get();
  //vector<Real>& rubatoStop = _rubatoStop.get();
  //int& rubatoNumber = _rubatoNumber.get();
  vector<Real>& bpmIntervals = _bpmIntervals.get();

  bpm = _pool.value<Real>("internal.bpm");
  ticks = _pool.value<vector<Real> >("internal.ticks");
  confidence = _pool.value<Real>("internal.confidence");
  estimates = _pool.value<vector<Real> >("internal.estimates");
  bpmIntervals = _pool.value<vector<Real> >("internal.bpmIntervals");
  //rubatoNumber = (int) _pool.value<Real>("internal.rubatoNumber");
  //try {
  //    rubatoStart = _pool.value<vector<Real> >("internal.rubatoStart");
  //    rubatoStop = _pool.value<vector<Real> >("internal.rubatoStop");
  //}
  //catch (EssentiaException&) {
  //  // no rubato regions then
  //}

}

void RhythmExtractor2013::reset() {
  _network->reset();
  _pool.remove("internal.ticks");
  _pool.remove("internal.confidence");
  _pool.remove("internal.bpm");
  _pool.remove("internal.estimates");
  _pool.remove("internal.bpmIntervals");
  //_pool.remove("internal.rubatoNumber");
  //try {
  //  _pool.remove("internal.rubatoStart");
  //  _pool.remove("internal.rubatoStop");
  //}
  //catch (EssentiaException&) {
  //  // were not in pool
  //}
}

} // namespace standard
} // namespace essentia
