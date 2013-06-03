#include "levelextractor.h"
#include "algorithmfactory.h"
#include "poolstorage.h"

using namespace std;

namespace essentia {
namespace streaming {


const char* LevelExtractor::name = "LevelExtractor";
const char* LevelExtractor::description = DOC("this algorithm extracts the loudness of an audio signal");

LevelExtractor::LevelExtractor() {

  declareInput(_signal, "signal", "the input audio signal");
  declareOutput(_loudnessValue, "loudness", "the loudness values");

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  _frameCutter = factory.create("FrameCutter",
                                "silentFrames", "noise",
                                "startFromZero", true);

  _loudness = factory.create("Loudness");

  _signal                        >>  _frameCutter->input("signal");
  _frameCutter->output("frame")  >>  _loudness->input("signal");
  _loudness->output("loudness")  >>  _loudnessValue;
}

void LevelExtractor::configure() {
  _frameCutter->configure(INHERIT("frameSize"),
                          INHERIT("hopSize"));
}

LevelExtractor::~LevelExtractor() {
  delete _frameCutter;
  delete _loudness;
}


} // namespace streaming
} // namespace essentia

namespace essentia {
namespace standard {

const char* LevelExtractor::name = "LevelExtractor";
const char* LevelExtractor::description = DOC("this algorithm extracts the loudness of an audio signal");

LevelExtractor::LevelExtractor() {
  declareInput(_signal, "signal", "the audio input signal");
  declareOutput(_loudness, "loudness", "the loudness values");

  createInnerNetwork();
}

LevelExtractor::~LevelExtractor() {
  delete _network;
}

void LevelExtractor::reset() {
  _network->reset();
  _pool.clear();
}

void LevelExtractor::configure() {
  _levelExtractor->configure(INHERIT("frameSize"), INHERIT("hopSize"));
}

void LevelExtractor::createInnerNetwork() {
  _levelExtractor = streaming::AlgorithmFactory::create("LevelExtractor");
  _vectorInput = new streaming::VectorInput<Real>();

  *_vectorInput                        >>  _levelExtractor->input("signal");
  _levelExtractor->output("loudness")  >>  PC(_pool, "internal.loudness");

  _network = new scheduler::Network(_vectorInput);
}


 void LevelExtractor::compute() {
  const vector<Real>& signal = _signal.get();
  _vectorInput->setVector(&signal);

  _network->run();

  vector<Real>& loudness = _loudness.get();

  loudness = _pool.value<vector<Real> >("internal.loudness");
}

} // namespace standard
} // namespace essentia

