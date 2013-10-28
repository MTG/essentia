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

#include "beattrackermultifeature.h"
#include "poolstorage.h"
#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* BeatTrackerMultiFeature::name = "BeatTrackerMultiFeature";
const char* BeatTrackerMultiFeature::description = DOC("This algorithm estimates the beat locations given an input signal. It computes a number of onset detection functions and estimates beat location candidates from them using TempoTapDegara algorithm. Thereafter the best candidates are selected using TempoTapMaxAgreement. The employed detection functions, and the optimal frame/hop sizes used for their computation are:\n"
"  - complex spectral difference (see 'complex' method in OnsetDetection algorithm, 2048/1024 with posterior x2 upsample or the detection function)\n"
"  - energy flux (see 'rms' method in OnsetDetection algorithm, the same settings)\n"
"  - spectral flux in Mel-frequency bands (see 'melflux' method in OnsetDetection algorithm, the same settings)\n"
"  - beat emphasis function (see 'beat_emphasis' method in OnsetDetectionGlobal algorithm, 2048/512)\n"
"  - spectral flux between histogrammed spectrum frames, measured by the modified information gain (see 'infogain' method in OnsetDetectionGlobal algorithm, 2048/512)\n"
"\n"
"You can follow these guidelines [2] to assess the quality of beats estimation based on the computed confidence value:\n"
"  - [0, 1)      very low confidence, the input signal is hard for the employed candidate beat trackers\n"
"  - [1, 1.5]    low confidence\n"
"  - (1.5, 3.5]  good confidence, accuracy around 80% in AMLt measure\n"
"  - (3.5, 5.32] excellent confidence\n"
"\n"
"Note that the algorithm requires the audio input with the 44100 Hz sampling rate in order to function correctly.\n"
"\n"
"References:\n"
"  [1] J. Zapata, M.E.P. Davies and E. Gómez, \"Multi Feature Beat tracker,\"\n"
"  submitted article to IEEE TSALP, 2013.\n"
"  [2] J.R. Zapata, A. Holzapfel, M.E.P. Davies, J.L. Oliveira, F. Gouyon,\n"
"  \"Assigning a confidence threshold on automatic beat annotation in large\n"
"  datasets\", International Society for Music Information Retrieval Conference\n"
"  (ISMIR'12), pp. 157-162, 2012\n");

//TODO update ref [1] when the article gets published

BeatTrackerMultiFeature::BeatTrackerMultiFeature() : AlgorithmComposite(),
    _frameCutter1(0), _windowing1(0), _fft1(0), _cart2polar1(0), _onsetRms1(0),
    _onsetComplex1(0), _ticksRms1(0), _ticksComplex1(0), _onsetMelFlux1(0),
    _ticksMelFlux1(0), _onsetBeatEmphasis3(0), _ticksBeatEmphasis3(0),
    _onsetInfogain4(0), _ticksInfogain4(0), _scale(0), _configured(false) {

  declareInput(_signal, 1024, "signal", "input signal");
  declareOutput(_ticks, 0, "ticks", "the estimated tick locations [s]");
  declareOutput(_confidence, "confidence", "confidence of the beat tracker [0, 5.32]");

  // NB: We want to have the same output stream type as in TempoTapTicks for
  // consistency. We need to increase buffer size of the output because the
  // algorithm works on the level of entire track and we need to push all values
  // in the output source at once.
  _ticks.setBufferType(BufferUsage::forLargeAudioStream);

}

void BeatTrackerMultiFeature::createInnerNetwork() {
  // internal algorithms
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter1         = factory.create("FrameCutter");
  _windowing1           = factory.create("Windowing");
  _fft1                 = factory.create("FFT");
  _cart2polar1          = factory.create("CartesianToPolar");
  _onsetRms1            = factory.create("OnsetDetection");
  _onsetComplex1        = factory.create("OnsetDetection");
  _onsetMelFlux1        = factory.create("OnsetDetection");
  _ticksRms1            = factory.create("TempoTapDegara");
  _ticksComplex1        = factory.create("TempoTapDegara");
  _ticksMelFlux1        = factory.create("TempoTapDegara");

  _onsetBeatEmphasis3   = factory.create("OnsetDetectionGlobal");
  _ticksBeatEmphasis3   = factory.create("TempoTapDegara");

  _onsetInfogain4       = factory.create("OnsetDetectionGlobal");
  _ticksInfogain4       = factory.create("TempoTapDegara");

  _tempoTapMaxAgreement = standard::AlgorithmFactory::create("TempoTapMaxAgreement");

  _scale = factory.create("Scale");
  // TODO this is a dummy algorithm (scale factor = 1.) used because SinkProxy
  // cannot be attached to multiple algorithms, but we need several processing
  // chains starting from the input.

  // Connect internal algorithms
  //_signal                                    >>   _frameCutter1->input("signal");
  _signal                                    >>   _scale->input("signal");
  _scale->output("signal")                   >>   _frameCutter1->input("signal");
  _frameCutter1->output("frame")             >>   _windowing1->input("frame");
  _windowing1->output("frame")               >>   _fft1->input("frame");
  _fft1->output("fft")                       >>   _cart2polar1->input("complex");
  _cart2polar1->output("magnitude")          >>   _onsetComplex1->input("spectrum");
  _cart2polar1->output("phase")              >>   _onsetComplex1->input("phase");
  _cart2polar1->output("magnitude")          >>   _onsetRms1->input("spectrum");
  _cart2polar1->output("phase")              >>   _onsetRms1->input("phase");
  _cart2polar1->output("magnitude")          >>   _onsetMelFlux1->input("spectrum");
  _cart2polar1->output("phase")              >>   _onsetMelFlux1->input("phase");

  _onsetComplex1->output("onsetDetection")   >>   _ticksComplex1->input("onsetDetections");
  _ticksComplex1->output("ticks")            >>   PC(_pool, "internal.ticksComplex");
  _onsetRms1->output("onsetDetection")       >>   _ticksRms1->input("onsetDetections");
  _ticksRms1->output("ticks")                >>   PC(_pool, "internal.ticksRms");
  _onsetMelFlux1->output("onsetDetection")   >>   _ticksMelFlux1->input("onsetDetections");
  _ticksMelFlux1->output("ticks")            >>   PC(_pool, "internal.ticksMelFlux");

  //_signal                                           >>   _onsetBeatEmphasis3->input("signal");
  _scale->output("signal")                         >>  _onsetBeatEmphasis3->input("signal");
  _onsetBeatEmphasis3->output("onsetDetections")   >>  _ticksBeatEmphasis3->input("onsetDetections");
  _ticksBeatEmphasis3->output("ticks")             >>  PC(_pool, "internal.ticksBeatEmphasis");

  //_signal                                           >> _onsetInfogain4->input("signal");
  _scale->output("signal")  >> _onsetInfogain4->input("signal");
  _onsetInfogain4->output("onsetDetections")        >> _ticksInfogain4->input("onsetDetections");
  _ticksInfogain4->output("ticks")                  >> PC(_pool, "internal.ticksInfogain");

  _network = new scheduler::Network(_scale);
}

void BeatTrackerMultiFeature::clearAlgos() {
  if (!_configured) return;

  delete _network;
  delete _tempoTapMaxAgreement;
}


BeatTrackerMultiFeature::~BeatTrackerMultiFeature() {
  clearAlgos();
}


void BeatTrackerMultiFeature::configure() {
  if (_configured) {
    clearAlgos();
  }

  _sampleRate = 44100.;
  // TODO will only work with _sampleRate = 44100, check what original
  // RhythmExtractor does with sampleRate parameter

  //_sampleRate   = parameter("sampleRate").toReal();
  createInnerNetwork();

  // Configure internal algorithms
  int minTempo = parameter("minTempo").toInt();
  int maxTempo = parameter("maxTempo").toInt();

  int frameSize1 = 2048;
  int hopSize1 = 1024;
  // NB: 2048/1024 frames followed by x2 resampling of OSD work better than
  // simply using 1024/512 frames with no resampling for 'complex', 'rms', and
  // 'melflux' onset detection function, according to the evaluation at MTG
  // (JZapata, DBogdanov)

  _scale->configure("factor", 1.);

  _frameCutter1->configure("frameSize", frameSize1,
                          "hopSize", hopSize1,
                          "silentFrames", "noise",
                          "startFromZero", true);

  _windowing1->configure("size", frameSize1, "type", "hann");
  _fft1->configure("size", frameSize1);
  _onsetComplex1->configure("method", "complex");
  _onsetRms1->configure("method", "rms");
  _onsetMelFlux1->configure("method", "melflux");
  _ticksComplex1->configure("sampleRateODF", _sampleRate/hopSize1,
                            "resample", "x2",
                            "minTempo", minTempo,
                            "maxTempo", maxTempo);
  _ticksRms1->configure("sampleRateODF", _sampleRate/hopSize1,
                        "resample", "x2",
                        "minTempo", minTempo,
                        "maxTempo", maxTempo);
  _ticksMelFlux1->configure("sampleRateODF",
                            _sampleRate/hopSize1,
                            "resample", "x2",
                            "minTempo", minTempo,
                            "maxTempo", maxTempo);

  int frameSize3 = 2048;
  int hopSize3 = 512;
  // NB: better than 2048/1024 plus x2 resampling according to evaluation (JZapata)
  // 2048/512 works better than 1024/512 for 'beat_emphasis' OSD according to
  // evaluation results (DBogdanov)
  _onsetBeatEmphasis3->configure("method", "beat_emphasis",
                                  "sampleRate", _sampleRate,
                                  "frameSize", frameSize3,
                                  "hopSize", hopSize3);
  _ticksBeatEmphasis3->configure("sampleRateODF", _sampleRate/hopSize3,
                                  "resample", "none",
                                  "minTempo", minTempo,
                                  "maxTempo", maxTempo);

  int frameSize4 = 2048;
  int hopSize4 = 512;
  // NB: 2048/512 performs better than 1024/512 accoding to evaluation (JZapata)
  _onsetInfogain4->configure("method", "infogain",
                             "sampleRate", _sampleRate,
                             "frameSize", frameSize4,
                             "hopSize", hopSize4);
  _ticksInfogain4->configure("sampleRateODF", _sampleRate/hopSize4,
                             "resample", "none",
                             "minTempo", minTempo,
                             "maxTempo", maxTempo);

  _configured = true;
}

AlgorithmStatus BeatTrackerMultiFeature::process() {
  if (!shouldStop()) return PASS;

  vector<vector<Real> > tickCandidates;
  vector<Real> ticks;
  Real confidence;

  tickCandidates.resize(5);

  // ticks candidates might be empty for very short signals, but
  // it is ok to feed empty tick vetors to TempoTapMaxAgreement
  if (_pool.contains<vector<Real> >("internal.ticksComplex")) {
    tickCandidates[0] = _pool.value<vector<Real> >("internal.ticksComplex");
  }
  if (_pool.contains<vector<Real> >("internal.ticksRms")) {
    tickCandidates[1] = _pool.value<vector<Real> >("internal.ticksRms");
  }
  if (_pool.contains<vector<Real> >("internal.ticksMelFlux")) {
    tickCandidates[2] = _pool.value<vector<Real> >("internal.ticksMelFlux");
  }
  if (_pool.contains<vector<Real> >("internal.ticksBeatEmphasis")) {
    tickCandidates[3] = _pool.value<vector<Real> >("internal.ticksBeatEmphasis");
  }
  if (_pool.contains<vector<Real> >("internal.ticksInfogain")) {
    tickCandidates[4] = _pool.value<vector<Real> >("internal.ticksInfogain");
  }

  _tempoTapMaxAgreement->input("tickCandidates").set(tickCandidates);
  _tempoTapMaxAgreement->output("ticks").set(ticks);
  _tempoTapMaxAgreement->output("confidence").set(confidence);
  _tempoTapMaxAgreement->compute();

  for (size_t i=0; i<ticks.size(); ++i) {
    _ticks.push(ticks[i]);
  }
  _confidence.push(confidence);
  return FINISHED;
}


void BeatTrackerMultiFeature::reset() {
  AlgorithmComposite::reset();
  _tempoTapMaxAgreement->reset();
}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* BeatTrackerMultiFeature::name = "BeatTrackerMultiFeature";
const char* BeatTrackerMultiFeature::description = DOC("This algorithm estimates the beat locations given an input signal. It computes a number of onset detection functions and estimates beat location candidates from them using TempoTapDegara algorithm. Thereafter the best candidates are selected using TempoTapMaxAgreement. The employed detection functions, and the optimal frame/hop sizes used for their computation are:\n"
"  - complex spectral difference (see 'complex' method in OnsetDetection algorithm, 2048/1024 with posterior x2 upsample or the detection function)\n"
"  - energy flux (see 'rms' method in OnsetDetection algorithm, the same settings)\n"
"  - spectral flux in Mel-frequency bands (see 'melflux' method in OnsetDetection algorithm, the same settings)\n"
"  - beat emphasis function (see 'beat_emphasis' method in OnsetDetectionGlobal algorithm, 2048/512)\n"
"  - spectral flux between histogrammed spectrum frames, measured by the modified information gain (see 'infogain' method in OnsetDetectionGlobal algorithm, 2048/512)\n"
"\n"
"You can follow these guidelines [2] to assess the quality of beats estimation based on the computed confidence value:\n"
"  - [0, 1)      very low confidence, the input signal is hard for the employed candidate beat trackers\n"
"  - [1, 1.5]    low confidence\n"
"  - (1.5, 3.5]  good confidence, accuracy around 80% in AMLt measure\n"
"  - (3.5, 5.32] excellent confidence\n"
"\n"
"Note that the algorithm requires the audio input with the 44100 Hz sampling rate in order to function correctly.\n"
"\n"
"References:\n"
"  [1] J. Zapata, M.E.P. Davies and E. Gómez, \"Multi Feature Beat tracker,\"\n"
"  submitted article to IEEE TSALP, 2013.\n"
"  [2] J.R. Zapata, A. Holzapfel, M.E.P. Davies, J.L. Oliveira, F. Gouyon,\n"
"  \"Assigning a confidence threshold on automatic beat annotation in large\n"
"  datasets\", International Society for Music Information Retrieval Conference\n"
"  (ISMIR'12), pp. 157-162, 2012\n");


BeatTrackerMultiFeature::BeatTrackerMultiFeature() {
  declareInput(_signal, "signal", "the audio input signal");
  declareOutput(_ticks, "ticks", " the estimated tick locations [s]");
  declareOutput(_confidence, "confidence", "confidence of the beat tracker [0, 5.32]");

  createInnerNetwork();
}

BeatTrackerMultiFeature::~BeatTrackerMultiFeature() {
  delete _network;
}

void BeatTrackerMultiFeature::configure() {
  _beatTracker->configure(//INHERIT("sampleRate"),
                          INHERIT("maxTempo"),
                          INHERIT("minTempo"));
}


void BeatTrackerMultiFeature::createInnerNetwork() {
  _beatTracker = streaming::AlgorithmFactory::create("BeatTrackerMultiFeature");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _beatTracker->input("signal");
  _beatTracker->output("ticks")  >>  PC(_pool, "internal.ticks");
  _beatTracker->output("confidence") >> PC(_pool, "internal.confidence");

  _network = new scheduler::Network(_vectorInput);
}

void BeatTrackerMultiFeature::compute() {
  // TODO: running this algorithm on consequent inputs always requires reset(),
  // which could be fixed by manually reseting here after computations are done
  const vector<Real>& signal = _signal.get();
  vector<Real>& ticks = _ticks.get();
  Real& confidence = _confidence.get();

  _vectorInput->setVector(&signal);
  _network->run();
  try {
    ticks = _pool.value<vector<Real> >("internal.ticks");
    confidence = _pool.value<Real> ("internal.confidence");
  }
  catch (EssentiaException&) {
    // no ticks were found because audio signal was too short
    ticks.clear();
    confidence = 0.;
  }
}

void BeatTrackerMultiFeature::reset() {
  _network->reset();
  _pool.remove("internal.ticks");
  _pool.remove("internal.confidence");
}

} // namespace standard
} // namespace essentia
