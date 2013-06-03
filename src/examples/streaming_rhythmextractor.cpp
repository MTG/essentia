#include <iostream>
#include "algorithmfactory.h"
#include "poolstorage.h"
#include "network.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

int main(int argc, char* argv[]) {
  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  if (argc != 2) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile" << endl;
    exit(1);
  }
  string audioFilename = argv[1];

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audioloader = factory.create("AudioLoader",
                                                   "filename", audioFilename);

  Algorithm* mono  = factory.create("MonoMixer");

  Algorithm* tempotap = factory.create("RhythmExtractor2013");
  tempotap->configure("method", "multifeature");  // best accuracy, but the largest computation time

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // audioloader -> mono
  connect(audioloader->output("audio"),      mono->input("audio"));
  connect(audioloader->output("numberChannels"),   mono->input("numberChannels"));
  connect(audioloader->output("sampleRate"), pool, "metadata.sampleRate");

  // mono -> tempotap
  connect(mono->output("audio"), tempotap->input("signal"));

  connect(tempotap->output("ticks"), pool, "rhythm.ticks");
  connect(tempotap->output("bpm"), pool, "rhythm.bpm");
  connect(tempotap->output("estimates"), pool, "rhythm.estimates");
  connect(tempotap->output("rubatoStart"), pool, "rhythm.rubatoStart");
  connect(tempotap->output("rubatoStop"), pool, "rhythm.rubatoStop");
  connect(tempotap->output("bpmIntervals"), pool, "rhythm.bpmIntervals");

  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  Network network(audioloader);
  network.run();


  // printing results
  cout << "-------- results --------" << endl;
  cout << "bpm: " << pool.value<Real>("rhythm.bpm") << endl;
  cout << "ticks: " << pool.value<vector<Real> >("rhythm.ticks") << endl;
  cout << "estimates: " << pool.value<vector<Real> >("rhythm.estimates") << endl;
  cout << "bpmIntervals: " << pool.value<vector<Real> >("rhythm.bpmIntervals") << endl;
  try {
      cout << "rubatoStart: " << pool.value<vector<Real> >("rhythm.rubatoStart") << endl;
      cout << "rubatoStop: " << pool.value<vector<Real> >("rhythm.rubatoStop") << endl;
  }
  catch (EssentiaException&) {
    cout << "No rubato regions found" << endl;
  }

  essentia::shutdown();

  return 0;
}
