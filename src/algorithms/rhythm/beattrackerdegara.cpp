/*
 * Copyright (C) 2006-2013 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "beattrackerdegara.h"
#include "poolstorage.h"
#include "algorithmfactory.h"

using namespace std;

namespace essentia {
namespace streaming {

const char* BeatTrackerDegara::name = "BeatTrackerDegara";
const char* BeatTrackerDegara::description = DOC("This algorithm estimates the beat locations given an input signal. It computes 'complex spectral difference' onset detection function and utilizes the beat tracking algorithm (TempoTapDegara) to extract beats [1]. The algorithm works with the optimized settings of 2048/1024 frame/hop size for the computation of the detection function, with its posterior x2 resampling.) While it has a lower accuracy than BeatTrackerMultifeature (see the evaluation results in [2]), its computational speed is significantly higher, which makes reasonable to apply this algorithm for batch processings of large amounts of audio signals.\n"
"\n"
"Note that the algorithm requires the audio input with the 44100 Hz sampling rate in order to function correctly.\n"
"\n"
"References:\n"
"  [1] N. Degara, E. A. Rua, A. Pena, S. Torres-Guijarro, M. E. Davies, and\n"
"  M. D. Plumbley, \"Reliability-informed beat tracking of musical signals,\"\n"
"  IEEE Transactions on Audio, Speech, and Language Processing, vol. 20,\n" 
"  no. 1, pp. 290–301, 2012.\n\n"
"  [2] J. Zapata, M.E.P. Davies and E. Gómez, \"Multi Feature Beat tracker,\"\n"
"  submitted article to IEEE TSALP, 2013.\n");

// TODO updated ref [2] when the article gets published

// evaluation results on a large collection of tracks [2] for essentia implementations (DBogdanov):
// *Degara:         65.4407   55.3735   45.6554   66.5149   45.6192   49.9258   69.5233   77.7232    2.2475   68.0827  -- fast
// *MultiFeature:   65.8871   53.9884   47.2754   66.4396   46.3123   50.8143   72.0530   80.5138    2.3865   67.5627  -- very slow
// RhythmExtractor: 49.6150   40.3908   21.4286   55.7790   29.8988   37.5160   41.1232   51.7029    1.5479   48.3133  -- slow


BeatTrackerDegara::BeatTrackerDegara() : AlgorithmComposite(),
    _frameCutter(0), _windowing(0), _fft(0), _cart2polar(0),
    _onsetComplex(0), _ticksComplex(0), _configured(false) {

  declareInput(_signal, 1024, "signal", "input signal");
  declareOutput(_ticks, 0, "ticks", "the estimated tick locations [s]");
}

void BeatTrackerDegara::createInnerNetwork() {
  // internal algorithms
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter         = factory.create("FrameCutter");
  _windowing           = factory.create("Windowing");
  _fft                 = factory.create("FFT");
  _cart2polar          = factory.create("CartesianToPolar");
  _onsetComplex        = factory.create("OnsetDetection");
  _ticksComplex        = factory.create("TempoTapDegara");

  // Connect internal algorithms
  _signal                                   >>   _frameCutter->input("signal");
  _frameCutter->output("frame")             >>   _windowing->input("frame");
  _windowing->output("frame")               >>   _fft->input("frame");
  _fft->output("fft")                       >>   _cart2polar->input("complex");
  _cart2polar->output("magnitude")          >>   _onsetComplex->input("spectrum");
  _cart2polar->output("phase")              >>   _onsetComplex->input("phase");
  _onsetComplex->output("onsetDetection")   >>   _ticksComplex->input("onsetDetections");
  _ticksComplex->output("ticks")            >>   _ticks;

  _network = new scheduler::Network(_frameCutter);
}

void BeatTrackerDegara::clearAlgos() {
  if (!_configured) return;
  delete _network;
}


BeatTrackerDegara::~BeatTrackerDegara() {
  clearAlgos();
}


void BeatTrackerDegara::configure() {
  if (_configured) {
    clearAlgos();
  }

  _sampleRate = 44100.;
  // TODO will only work with _sampleRate = 44100, check what original
  // RhythmExtractor does with sampleRate parameter

  //_sampleRate   = parameter("sampleRate").toReal();
  createInnerNetwork();

  // Configure internal algorithms

  int frameSize = 2048;
  int hopSize = 1024;
  // NB: 2048/1024 frames followed by x2 resampling of OSD work better than
  // simply using 1024/512 frames with no resampling for 'complex'  onset
  // detection function, according to the evaluation at MTG (JZapata, DBogdanov)

  _frameCutter->configure("frameSize", frameSize,
                          "hopSize", hopSize,
                          "silentFrames", "noise",
                          "startFromZero", true);

  _windowing->configure("size", frameSize, "type", "hann");
  _fft->configure("size", frameSize);
  _onsetComplex->configure("method", "complex");
  _ticksComplex->configure("sampleRateODF", _sampleRate/hopSize,
                            "resample", "x2",
                            "minTempo", parameter("minTempo").toInt(),
                            "maxTempo", parameter("maxTempo").toInt());
  _configured = true;
}


void BeatTrackerDegara::reset() {
  AlgorithmComposite::reset();
}

} // namespace streaming
} // namespace essentia



namespace essentia {
namespace standard {

const char* BeatTrackerDegara::name = "BeatTrackerDegara";
const char* BeatTrackerDegara::description = DOC("This algorithm estimates the beat locations given an input signal. It computes 'complex spectral difference' onset detection function and utilizes the beat tracking algorithm (TempoTapDegara) to extract beats [1]. The algorithm works with the optimized settings of 2048/1024 frame/hop size for the computation of the detection function, with its posterior x2 resampling.) While it has a lower accuracy than BeatTrackerMultifeature (see the evaluation results in [2]), its computational speed is significantly higher, which makes reasonable to apply this algorithm for batch processings of large amounts of audio signals.\n"
"\n"
"Note that the algorithm requires the audio input with the 44100 Hz sampling rate in order to function correctly.\n"
"\n"
"References:\n"
"  [1] N. Degara, E. A. Rua, A. Pena, S. Torres-Guijarro, M. E. Davies, and\n"
"  M. D. Plumbley, \"Reliability-informed beat tracking of musical signals,\"\n"
"  IEEE Transactions on Audio, Speech, and Language Processing, vol. 20,\n" 
"  no. 1, pp. 290–301, 2012.\n\n"
"  [2] J. Zapata, M.E.P. Davies and E. Gómez, \"Multi Feature Beat tracker,\"\n"
"  submitted article to IEEE TSALP, 2013.\n");

// TODO updated ref [2] when the article gets published


BeatTrackerDegara::BeatTrackerDegara() {
  declareInput(_signal, "signal", "the audio input signal");
  declareOutput(_ticks, "ticks", " the estimated tick locations [s]");

  createInnerNetwork();
}

BeatTrackerDegara::~BeatTrackerDegara() {
  delete _network;
}

void BeatTrackerDegara::configure() {
  _beatTracker->configure(//INHERIT("sampleRate"),
                          INHERIT("maxTempo"),
                          INHERIT("minTempo"));
}


void BeatTrackerDegara::createInnerNetwork() {
  _beatTracker = streaming::AlgorithmFactory::create("BeatTrackerDegara");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput  >>  _beatTracker->input("signal");
  _beatTracker->output("ticks")  >>  PC(_pool, "internal.ticks");

  _network = new scheduler::Network(_vectorInput);
}

void BeatTrackerDegara::compute() {
  // TODO: running this algorithm on consequent inputs always requires reset(),
  // which could be fixed by manually reseting here after computations are done
  const vector<Real>& signal = _signal.get();
  vector<Real>& ticks = _ticks.get();

  _vectorInput->setVector(&signal);
  _network->run();
  try {
    ticks = _pool.value<vector<Real> >("internal.ticks");
  }
  catch (EssentiaException&) {
    // no ticks were found because audio signal was too short
    ticks.clear();
  }
}

void BeatTrackerDegara::reset() {
  _network->reset();
  _pool.remove("internal.ticks");
}

} // namespace standard
} // namespace essentia
