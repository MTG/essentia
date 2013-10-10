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

#include "rhythmdescriptors.h"
#include "algorithmfactory.h"
#include "network.h"
#include "poolstorage.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* RhythmDescriptors::name = "RhythmDescriptors";
const char* RhythmDescriptors::description = DOC("This algorithm computes rhythm features. It combines RhythmExtractor2013 for beat tracking and BPM estimation with BpmHistogramDescriptors algorithms.");

RhythmDescriptors::RhythmDescriptors() {
  _configured = false;

  declareInput(_signal, "signal", "the input audio signal");

  declareOutput(_ticks, "beats_position", "See RhythmExtractor2013 algorithm documentation");
  declareOutput(_bpm, "bpm", "See RhythmExtractor2013 algorithm documentation");
  declareOutput(_estimates, "bpm_estimates", "See RhythmExtractor2013 algorithm documentation");
  declareOutput(_bpmIntervals, "bpm_intervals", "See RhythmExtractor2013 algorithm documentation");
  //FIXME we need better rubato estimation algorithm
  //declareOutput(_rubatoStart, "rubato_start", "See RhythmExtractor2013 algorithm documentation");
  //declareOutput(_rubatoStop, "rubato_stop", "See RhythmExtractor2013 algorithm documentation");
  //declareOutput(_rubatoNumber, "rubato_number", "See RhythmExtractor2013 algorithm documentation");

  declareOutput(_firstPeakBPM, "first_peak_bpm", "See BpmHistogramDescriptors algorithm documentation");
  declareOutput(_firstPeakSpread, "first_peak_spread", "See BpmHistogramDescriptors algorithm documentation");
  declareOutput(_firstPeakWeight, "first_peak_weight", "See BpmHistogramDescriptors algorithm documentation");
  declareOutput(_secondPeakBPM, "second_peak_bpm", "See BpmHistogramDescriptors algorithm documentation");
  declareOutput(_secondPeakSpread, "second_peak_spread", "See BpmHistogramDescriptors algorithm documentation");
  declareOutput(_secondPeakWeight, "second_peak_weight", "See BpmHistogramDescriptors algorithm documentation");
}


void RhythmDescriptors::createInnerNetwork() {
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _bpmHistogramDescriptors = factory.create("BpmHistogramDescriptors");
  _rhythmExtractor = factory.create("RhythmExtractor2013");

  _signal >> _rhythmExtractor->input("signal");

  _rhythmExtractor->output("ticks") >> PC(_pool, "internal.ticks");
  _rhythmExtractor->output("bpm") >> PC(_pool, "internal.bpm");
  _rhythmExtractor->output("estimates")     >> PC(_pool, "internal.estimates");
  _rhythmExtractor->output("bpmIntervals")  >> PC(_pool, "internal.bpmIntervals");
  //_rhythmExtractor->output("rubatoStart")   >> PC(_pool, "internal.rubatoStart");
  //_rhythmExtractor->output("rubatoStop")    >> PC(_pool, "internal.rubatoStop");
  //_rhythmExtractor->output("rubatoNumber")    >> PC(_pool, "internal.rubatoNumber");

  // TODO current limitation: seems like if one output of an algo is connected to
  // SourceProxy, other outputs directed somewhere else will be empty. Bug?
  // Temporal workaround: use pool to store all data instead of routing directly
  // to the source proxies.

  _rhythmExtractor->output("bpmIntervals")  >> _bpmHistogramDescriptors->input("bpmIntervals");
  _bpmHistogramDescriptors->output("firstPeakBPM")      >> _firstPeakBPM;
  _bpmHistogramDescriptors->output("firstPeakSpread")   >> _firstPeakSpread;
  _bpmHistogramDescriptors->output("firstPeakWeight")   >> _firstPeakWeight;
  _bpmHistogramDescriptors->output("secondPeakBPM")     >> _secondPeakBPM;
  _bpmHistogramDescriptors->output("secondPeakSpread")  >> _secondPeakSpread;
  _bpmHistogramDescriptors->output("secondPeakWeight")  >> _secondPeakWeight;

  _network = new scheduler::Network(_rhythmExtractor);
}


void RhythmDescriptors::configure() {
  if (_configured) {
    clearAlgos();
  }
  createInnerNetwork();
  _configured = true;
}


void RhythmDescriptors::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


AlgorithmStatus RhythmDescriptors::process() {
  if (!shouldStop()) return PASS;

  _bpm.push(_pool.value<Real>("internal.bpm"));
  _ticks.push(_pool.value<vector<Real> >("internal.ticks"));
  _estimates.push(_pool.value<vector<Real> >("internal.estimates"));
  _bpmIntervals.push(_pool.value<vector<Real> >("internal.bpmIntervals"));
  //_rubatoStart.push(_pool.value<vector<Real> >("internal.rubatoStart"));
  //_rubatoStop.push(_pool.value<vector<Real> >("internal.rubatoStop"));
  //_rubatoNumber.push((int) _pool.value<Real>("internal.rubatoStop"));

  return FINISHED;
}


RhythmDescriptors::~RhythmDescriptors() {
  clearAlgos();
}


void RhythmDescriptors::reset() {
    AlgorithmComposite::reset();
}

} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

const char* RhythmDescriptors::name = "RhythmDescriptors";
const char* RhythmDescriptors::description = DOC("this algorithm computes low level rhythm features");

RhythmDescriptors::RhythmDescriptors() {
  declareInput(_signal, "signal", "the audio input signal");
  declareOutput(_ticks,           "beats_position", "See RhythmExtractor2013 algorithm documentation");
  declareOutput(_bpm,             "bpm", "See RhythmExtractor2013 algorithm documentation");
  declareOutput(_estimates,       "bpm_estimates", "See RhythmExtractor2013 algorithm documentation");
  declareOutput(_bpmIntervals,    "bpm_intervals", "See RhythmExtractor2013 algorithm documentation");
  //declareOutput(_rubatoStart,     "rubato_start", "See RhythmExtractor2013 algorithm documentation");
  //declareOutput(_rubatoStop,      "rubato_stop", "See RhythmExtractor2013 algorithm documentation");
  //declareOutput(_rubatoNumber,    "rubato_number", "See RhythmExtractor2013 algorithm documentation");

  declareOutput(_firstPeakBPM,    "first_peak_bpm", "See BpmHistogramDescriptors algorithm documentation");
  declareOutput(_firstPeakSpread, "first_peak_spread", "See BpmHistogramDescriptors algorithm documentation");
  declareOutput(_firstPeakWeight, "first_peak_weight", "See BpmHistogramDescriptors algorithm documentation");
  declareOutput(_secondPeakBPM,   "second_peak_bpm", "See BpmHistogramDescriptors algorithm documentation");
  declareOutput(_secondPeakSpread,"second_peak_spread", "See BpmHistogramDescriptors algorithm documentation");
  declareOutput(_secondPeakWeight,"second_peak_weight", "See BpmHistogramDescriptors algorithm documentation");

  createInnerNetwork();
}

RhythmDescriptors::~RhythmDescriptors() {
  delete _network;
}

void RhythmDescriptors::reset() {
  _network->reset();
}

void RhythmDescriptors::configure() {
  _rhythmDescriptors->configure();
}

void RhythmDescriptors::createInnerNetwork() {
  _rhythmDescriptors = streaming::AlgorithmFactory::create("RhythmDescriptors");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _rhythmDescriptors->input("signal");

  _rhythmDescriptors->output("beats_position")      >>  PC(_pool, "beats_position");
  _rhythmDescriptors->output("bpm")                 >>  PC(_pool, "bpm");
  _rhythmDescriptors->output("bpm_estimates")       >>  PC(_pool, "bpm_estimates");
  _rhythmDescriptors->output("bpm_intervals")       >>  PC(_pool, "bpm_intervals");
  //_rhythmDescriptors->output("rubato_start")        >>  PC(_pool, "rubato_start");
  //_rhythmDescriptors->output("rubato_stop")         >>  PC(_pool, "rubato_stop");
  //_rhythmDescriptors->output("rubato_number")       >>  PC(_pool, "rubato_number");

  _rhythmDescriptors->output("first_peak_bpm")      >>  PC(_pool, "first_peak_bpm");
  _rhythmDescriptors->output("first_peak_spread")   >>  PC(_pool, "first_peak_spread");
  _rhythmDescriptors->output("first_peak_weight")   >>  PC(_pool, "first_peak_weight");
  _rhythmDescriptors->output("second_peak_bpm")     >>  PC(_pool, "second_peak_bpm");
  _rhythmDescriptors->output("second_peak_spread")  >>  PC(_pool, "second_peak_spread");
  _rhythmDescriptors->output("second_peak_weight")  >>  PC(_pool, "second_peak_weight");

  _network = new scheduler::Network(_vectorInput);
}

void RhythmDescriptors::compute() {
  const vector<Real>& signal = _signal.get();
  _vectorInput->setVector(&signal);

  _network->run();

  _bpm.get()          = _pool.value<Real>("bpm");
  _ticks.get()        = _pool.value<vector<Real> >("beats_position");
  _estimates.get()    = _pool.value<vector<Real> >("bpm_estimates");
  //_bpmIntervals.get() = _pool.value<vector<Real> >("bpm_intervals");
  //_rubatoNumber.get() = (int) _pool.value<Real>("rubato_number");
  //try {
  //_rubatoStart.get()  = _pool.value<vector<Real> >("rubato_start");
  //_rubatoStop.get()   = _pool.value<vector<Real> >("rubato_stop");
  //}
  //catch (EssentiaException &) { // no rubato regions found
  //  _rubatoStart.get() = vector<Real>();
  //  _rubatoStop.get() = vector<Real>();
  //}

  _firstPeakBPM.get()     = _pool.value<vector<Real> >("first_peak_bpm")[0];
  _firstPeakSpread.get()  = _pool.value<vector<Real> >("first_peak_spread")[0];
  _firstPeakWeight.get()  = _pool.value<vector<Real> >("first_peak_weight")[0];
  _secondPeakBPM.get()    = _pool.value<vector<Real> >("second_peak_bpm")[0];
  _secondPeakSpread.get() = _pool.value<vector<Real> >("second_peak_spread")[0];
  _secondPeakWeight.get() = _pool.value<vector<Real> >("second_peak_weight")[0];
}

} // namespace standard
} // namespace essentia

