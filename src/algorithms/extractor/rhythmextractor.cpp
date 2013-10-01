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
#include "rhythmextractor.h"
#include "tnt/tnt.h"
#include "essentiamath.h"
#include "poolstorage.h"
#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* RhythmExtractor::name = "RhythmExtractor";
const char* RhythmExtractor::description = DOC("This algorithm estimates the tempo in bpm from an input signal, as well as the beat locations. It combines TempoTap and TempoTapTicks.\n"
"\n"
"Note that this algorithm is outdated in terms of beat tracking accuracy, and it is highly recommended to use RhythmExtractor2013 instead.\n"
"\n"
"Quality: outdated (use RhythmExtractor2013 instead).\n"
"\n"
"An exception is thrown if neither \"useOnset\" nor \"useBands\" are enabled (i.e. set to true).");

inline Real lagToBpm(Real lag, Real sampleRate, Real hopSize) {
  return 60.0 * sampleRate / lag / hopSize;
}

inline Real bpmToLag(Real bpm, Real sampleRate, Real hopSize) {
  return lagToBpm(bpm, sampleRate, hopSize);
}

RhythmExtractor::RhythmExtractor()
  : AlgorithmComposite(), _frameCutter(0), _windowing(0), _fft(0), _cart2polar(0),
    _onsetHfc(0), _onsetComplex(0), _spectrum(0), _tempoTapBands(0), _tempoScaleBands(0),
    _tempoTap(0), _tempoTapTicks(0), /*_bpmRubato(0),*/ _multiplexer(0), _startStopSilence(0),
    _derivative(0), _max(0), _configured(false) {

  _preferredBufferSize = 1024;
  declareInput(_signal, _preferredBufferSize, "signal", "input signal");

  declareOutput(_bpm, 0, "bpm", "the tempo estimation [bpm]");
  declareOutput(_ticks, 0, "ticks", " the estimated tick locations [s]");
  declareOutput(_estimates, 0, "estimates", "the bpm estimation per frame [bpm]");
  //TODO we need better rubato estimation algorithm
  //declareOutput(_rubatoStart, 0, "rubatoStart", "list of start times for rubato section [s]");
  //declareOutput(_rubatoStop, 0, "rubatoStop", "list of stop times of rubato section [s]");
  //declareOutput(_rubatoNumber, 0, "rubatoNumber", "number of rubato section [s]");
  declareOutput(_bpmIntervals, 0, "bpmIntervals", "list of beats interval [s]");
}

void RhythmExtractor::createInnerNetwork() {

  //streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter      = factory.create("FrameCutter");
  _windowing        = factory.create("Windowing");
  _tempoTap         = factory.create("TempoTap");
  _tempoTapTicks    = factory.create("TempoTapTicks");
  _startStopSilence = factory.create("StartStopSilence");
  //_bpmRubato        = standard::AlgorithmFactory::create("BpmRubato");

  // Connect internal algorithms
  _signal                                  >>  _frameCutter->input("signal");
  _frameCutter->output("frame")            >>  _windowing->input("frame");
  _frameCutter->output("frame")            >>  _startStopSilence->input("frame");
  _startStopSilence->output("startFrame")  >>  PC(_pool, "internal.startSilence");
  _startStopSilence->output("stopFrame")   >>  PC(_pool, "internal.stopSilence");


  if (_useOnset && _useBands) {
    _multiplexer = factory.create("Multiplexer",
                                  "numberRealInputs", 3,
                                  "numberVectorRealInputs", 1);
  }
  else if (_useOnset) {
    _multiplexer = factory.create("Multiplexer",
                                  "numberRealInputs", 3);
  }
  else {
    _multiplexer = factory.create("Multiplexer",
                                  "numberVectorRealInputs", 1);
  }

  if (_useOnset) {
    // create algos:
    _fft           = factory.create("FFT");
    _cart2polar    = factory.create("CartesianToPolar");
    _onsetHfc      = factory.create("OnsetDetection");
    _onsetComplex  = factory.create("OnsetDetection");
    _derivative    = factory.create("Derivative");
    _max           = factory.create("Clipper", "min", 0.0, "max", numeric_limits<Real>::max());

    // connect algos:
    _windowing->output("frame")              >>  _fft->input("frame");

    _fft->output("fft")                      >>  _cart2polar->input("complex");

    _cart2polar->output("magnitude")         >>  _onsetHfc->input("spectrum");
    _cart2polar->output("phase")             >>  _onsetHfc->input("phase");

    _cart2polar->output("magnitude")         >>  _onsetComplex->input("spectrum");
    _cart2polar->output("phase")             >>  _onsetComplex->input("phase");

    _onsetHfc->output("onsetDetection")      >>  _multiplexer->input("real_0");

    _onsetHfc->output("onsetDetection")      >>  _derivative->input("signal");
    _derivative->output("signal")            >>  _max->input("signal");
    _max->output("signal")                   >>  _multiplexer->input("real_1");

    _onsetComplex->output("onsetDetection")  >>  _multiplexer->input("real_2");
  }

  if (_useBands) {
    _spectrum         = factory.create("Spectrum");
    _tempoTapBands    = factory.create("FrequencyBands");
    _tempoScaleBands  = factory.create("TempoScaleBands");

    // for useBands = True && useOnset = False, faster than FFT+cart2polar
    _windowing->output("frame")                  >>  _spectrum->input("frame");

    _spectrum->output("spectrum")                >>  _tempoTapBands->input("spectrum");

    _tempoTapBands->output("bands")              >>  _tempoScaleBands->input("bands");

    _tempoScaleBands->output("cumulativeBands")  >>  NOWHERE;
    _tempoScaleBands->output("scaledBands")      >>  _multiplexer->input("vector_0");
  }

  // the vector of features is the output of the multiplexer.
  _multiplexer->output("data")               >>  _tempoTap->input("featuresFrame");

  _tempoTap->output("periods")               >>  _tempoTapTicks->input("periods");
  _tempoTap->output("phases")                >>  _tempoTapTicks->input("phases");

  _tempoTapTicks->output("matchingPeriods")  >>  PC(_pool, "internal.matchingPeriods");
  _tempoTapTicks->output("ticks")            >>  PC(_pool, "internal.ticks");


  _network = new scheduler::Network(_frameCutter);
}

void RhythmExtractor::clearAlgos() {
  if (!_configured) return;
  // it is safe to call this function here, as the inner network isn't connected to
  // anything outside, so it won't propagate and try to delete stuff twice
  delete _network;
  //delete _bpmRubato;
}


RhythmExtractor::~RhythmExtractor() {
  clearAlgos();
}


void RhythmExtractor::configure() {
  if (_configured) {
    clearAlgos();
  }

  _sampleRate   = parameter("sampleRate").toReal();
  _numberFrames = parameter("numberFrames").toInt();
  _frameHop     = parameter("frameHop").toInt();
  _frameSize    = parameter("frameSize").toInt();
  _hopSize      = parameter("hopSize").toInt();
  _zeroPadding = 0;
  _periodTolerance = 5.;
  Real bandsFreq[] = {40.0, 413.16, 974.51, 1818.94, 3089.19, 5000.0, 7874.4, 12198.29, 17181.13};
  Real bandsGain[] = {2.0, 3.0, 2.0, 1.0, 1.2, 2.0, 3.0, 2.5};
  _frameTime = _hopSize / _sampleRate;
  _useOnset = parameter("useOnset").toBool();
  _useBands = parameter("useBands").toBool();
  _tolerance = parameter("tolerance").toReal();
  _lastBeatInterval = parameter("lastBeatInterval").toReal();

  if (!_useOnset && !_useBands) {
    throw EssentiaException("RhythmExtractor: No input features selected.");
  }


  createInnerNetwork();



  // Configure internal algorithms
  _frameCutter->configure("frameSize", _frameSize,
                          "hopSize", _hopSize,
                          "silentFrames", "noise",
                          "startFromZero", false);

  _windowing->configure("size", _frameSize,
                        "zeroPadding", _zeroPadding);
  //                    "type", "blackmanharris62");

  if (_useOnset) {
    _fft->configure("size", _frameSize);
    _onsetHfc->configure("method", "hfc",
                         "sampleRate", _sampleRate);
    _onsetComplex->configure("method", "complex",
                             "sampleRate", _sampleRate);
  }

  if (_useBands) {
    _spectrum->configure("size", _frameSize);
    _tempoTapBands->configure("frequencyBands", arrayToVector<Real>(bandsFreq));
    _tempoScaleBands->configure("bandsGain", arrayToVector<Real>(bandsGain));
  }

  _tempoTap->configure("sampleRate", _sampleRate,
                       "numberFrames", _numberFrames,
                       "frameHop", _frameHop,
                       "frameSize", _hopSize,
                       "tempoHints", parameter("tempoHints").toVectorReal(),
                       "minTempo", parameter("minTempo").toInt(),
                       "maxTempo", parameter("maxTempo").toInt());

  _tempoTapTicks->configure("hopSize", _hopSize,
                            "frameHop", _frameHop,
                            "sampleRate", _sampleRate);
  _configured = true;

}

AlgorithmStatus RhythmExtractor::process() {
  if (!shouldStop()) return PASS;

  Real framesPerSecond = _sampleRate / _hopSize;
  int nframes = _frameCutter->output("frame").totalProduced();
  Real fileLength = nframes / framesPerSecond;

  Real startSilence = _pool.value<Real>("internal.startSilence");
  Real stopSilence = _pool.value<Real>("internal.stopSilence");

  // stop silence should be maximum at 5 seconds before the end of the file
  // FIXME: why at 5 seconds? MAGIG NUMBER!!
  stopSilence = max(stopSilence, nframes - (Real)5.*framesPerSecond);

  // if the file was too short, fill the rest of the buffer with zeros
  // compute the beat candidates on the last incomplete buffer
  vector<Real> empty;
  //
  // onsets (3 values) + bands(8 values)
  if (_useOnset && _useBands)  empty.resize(11);
  else if (_useOnset && !_useBands)  empty.resize(3);
  else  empty.resize(8);

  // FIXME: probably the correct fix would be to have the inner network split in half, and then schedule
  //        someone in between to inject those empty frames
  /*
  while (nframes % _frameHop != 0) {
    cout << "mplex acquire" << endl;
    _multiplexer->output("data").push(empty);
    cout << "mplex acquire ok" << endl;
    // FIXME: we should have a way to do runConsumerTasks explcitly, because here it makes
    //        sense semantically to do so, as we want to feed empty frames first
    // at the moment, we have a workaround that consists in having a bigger output buffer
    // so it doesn't get full before we even start doing useful stuff
    //runConsumerTasksNow(_multiplexer->output("data"));
    nframes++;
  }
  */

  const vector<vector<Real> >& list_ticks = _pool.value<vector<vector<Real> > >("internal.ticks");
  vector<Real> ticks;
  for (int i = 0; i < (int)list_ticks.size(); i++) {
    for (int j = 0; j < (int)list_ticks[i].size(); j++) {
      ticks.push_back(list_ticks[i][j]);
    }
  }
  // compute the last ticks and prune the ones found after the end of file
  // eaylon: doing this may lead to have more ticks than there are in the real file.
  if (ticks.size() > 1) {
    // fill up to end of file
    if (fileLength > ticks[ticks.size()-1]) {
      Real lastPeriod = ticks[ticks.size()-1] - ticks[ticks.size()-2];
      while ( ticks[ticks.size()-1] + lastPeriod < fileLength - _lastBeatInterval ) {
        //std::cout << ticks[ticks.size()-1] + lastPeriod << " " << fileLength - _lastBeatInterval << std::endl;
        // if ( ticks[ticks.size()-1] > fileLength - _lastBeatInterval ) { // eaylon: this is redundant
        //   break;
        // }
        ticks.push_back(ticks[ticks.size()-1] + lastPeriod);
      }
    }

    for (int i = 0; i < (int)ticks.size(); i++) {
      // remove all negative ticks
      if (ticks[i] < startSilence / framesPerSecond ) {
        ticks.erase(ticks.begin() + i);
        i--;
      }
    }
    for (int i = 0; i < (int)ticks.size(); i++) {
      // kill all ticks from 350ms before the end of start
      if (ticks[i] > stopSilence / framesPerSecond - _lastBeatInterval ) {
        ticks.erase(ticks.begin() + i);
        i--;
      }
    }

    for (int i = 1; i < (int)ticks.size(); i++) {
      // prune all beats closer than tolerance
      if (ticks[i] - ticks[i-1] < _tolerance) {
        ticks.erase(ticks.begin() + i);
        i--;
      }
    }

    // prune all beats doing a backward off beat
    // WARNING: MAGIC NUMER: 0.100
    for (int i=3; i < int(ticks.size()); ++i) {
      if ((abs((ticks[i] - ticks[i-2]) - 1.5*(ticks[i] - ticks[i-1])) < 0.100) &&
          (abs(ticks[i] - ticks[i-1] - ticks[i-2] + ticks[i-3]) < 0.100)) {
        ticks.erase(ticks.begin() + i - 2);
        i--;
      }
    }
  }
  _ticks.push(ticks);

  const vector<vector<Real> >& matchingPeriods = _pool.value<vector<vector<Real> > >("internal.matchingPeriods");
  vector<Real> bpmEstimateList;

  for (int i=0; i < int(matchingPeriods.size()); ++i) {
    for (int j=0; j < int(matchingPeriods[i].size()); ++j) {
      if (matchingPeriods[i][j] != 0) {
        Real periodEstimate = matchingPeriods[i][j];
        bpmEstimateList.push_back(lagToBpm(periodEstimate, _sampleRate, _hopSize));
      }
    }
  }

  vector<Real> estimates;
  Real bpm;
  if (bpmEstimateList.size() > 0) {
    Real closestBpm = 0;
    vector<Real> countedBins;
    for (int i=0; i < int(bpmEstimateList.size()); ++i) {
      bpmEstimateList[i] /= 2.;
    }
    bincount(bpmEstimateList, countedBins);
    closestBpm = argmax(countedBins) * 2;
    for (int i=0; i < int(bpmEstimateList.size()); ++i) {
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

  _estimates.push(estimates);
  _bpm.push(bpm);

  vector<Real> bpmIntervals;
  if (ticks.size() > 1) {
    // computing beats intervals
    bpmIntervals.resize(ticks.size() - 1);
    for (int i = 1; i < (int)ticks.size(); i++) {
      bpmIntervals[i-1] = ticks[i] - ticks[i-1];
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

  _bpmIntervals.push(bpmIntervals);

  return FINISHED;
}


void RhythmExtractor::reset() {
  AlgorithmComposite::reset();
  //_bpmRubato->reset();
}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* RhythmExtractor::name = "RhythmExtractor";
const char* RhythmExtractor::description = DOC("This algorithm estimates the tempo in bpm from an input signal, as well as the beat locations. It combines TempoTap and TempoTapTicks.\n"
"\n"
"Note that this algorithm is outdated in terms of beat tracking accuracy, and it is highly recommended to use RhythmExtractor2013 instead.\n"
"\n"
"Quality: outdated (use RhythmExtractor2013 instead).\n"
"\n"
"An exception is thrown if neither \"useOnset\" nor \"useBands\" are enabled (i.e. set to true).");

RhythmExtractor::RhythmExtractor() {
  declareInput(_signal, "signal", "the audio input signal");
  declareOutput(_bpm, "bpm", "the tempo estimation [bpm]");
  declareOutput(_ticks, "ticks", " the estimated tick locations [s]");
  declareOutput(_estimates, "estimates", "the bpm estimation per frame [bpm]");
  //declareOutput(_rubatoStart, "rubatoStart", "list of start times for rubato section [s]");
  //declareOutput(_rubatoStop, "rubatoStop", "list of stop times of rubato section [s]");
  //declareOutput(_rubatoNumber, "rubatoNumber", "number of rubato section");
  declareOutput(_bpmIntervals, "bpmIntervals", "list of beats interval [s]");

  createInnerNetwork();
}

RhythmExtractor::~RhythmExtractor() {
  delete _network;
}

void RhythmExtractor::configure() {
  _rhythmExtractor->configure(INHERIT("useOnset"),     INHERIT("useBands"),
                              INHERIT("hopSize"),      INHERIT("frameSize"),
                              INHERIT("numberFrames"), INHERIT("frameHop"),
                              INHERIT("sampleRate"),   INHERIT("tolerance"),
                              INHERIT("tempoHints"),   INHERIT("maxTempo"),
                              INHERIT("minTempo"),     INHERIT("lastBeatInterval"));
}


void RhythmExtractor::createInnerNetwork() {
  _rhythmExtractor = streaming::AlgorithmFactory::create("RhythmExtractor");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _rhythmExtractor->input("signal");
  _rhythmExtractor->output("ticks")         >>  PC(_pool, "internal.ticks");
  _rhythmExtractor->output("bpm")           >>  PC(_pool, "internal.bpm");
  _rhythmExtractor->output("estimates")     >>  PC(_pool, "internal.estimates");
  //_rhythmExtractor->output("rubatoStart")   >>  PC(_pool, "internal.rubatoStart");
  //_rhythmExtractor->output("rubatoStop")    >>  PC(_pool, "internal.rubatoStop");
  //_rhythmExtractor->output("rubatoNumber")    >>  PC(_pool, "internal.rubatoNumber");
  _rhythmExtractor->output("bpmIntervals")  >>  PC(_pool, "internal.bpmIntervals");

  _network = new scheduler::Network(_vectorInput);
}

void RhythmExtractor::compute() {
  const vector<Real>& signal = _signal.get();
  _vectorInput->setVector(&signal);

  _network->run();


  Real& bpm = _bpm.get();
  vector<Real>& ticks = _ticks.get();
  vector<Real>& estimates = _estimates.get();
  //vector<Real>& rubatoStart = _rubatoStart.get();
  //vector<Real>& rubatoStop = _rubatoStop.get();
  //int& rubatoNumber = _rubatoNumber.get();
  vector<Real>& bpmIntervals = _bpmIntervals.get();

  bpm = _pool.value<Real>("internal.bpm");
  ticks = _pool.value<vector<Real> >("internal.ticks");
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


void RhythmExtractor::reset() {
  _network->reset();
  _pool.remove("internal.ticks");
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
