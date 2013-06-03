#include <iostream>
#include <fstream>
#include "algorithmfactory.h"
#include "poolstorage.h"
#include "vectorinput.h"
#include "network.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

int main(int argc, char* argv[]) {

  if (argc != 4) {
    cout << "ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input beats_list_file output_file" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string beatsFilename = argv[2];
  string outputFilename = argv[3];

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////
  Real sampleRate = 44100.0;

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();


  ifstream inbeats(beatsFilename.c_str());
  vector<Real> beats;
  while (true) {
    Real beat; inbeats >> beat;
    if (inbeats.eof()) break;
    beats.push_back(beat);
  }
    cout << "beats: " << beats << endl;

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* beatsMarker = factory.create("AudioOnsetsMarker",
                                          "type", "beep",
                                          "onsets", beats);

  Algorithm* writer = factory.create("MonoWriter",
                                     "filename", outputFilename);

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // Audio -> FrameCutter
  connect(audio->output("audio"), beatsMarker->input("signal"));
  connect(beatsMarker->output("signal"), writer->input("audio"));



  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  Network network(audio);
  network.run();

  //essentia::shutdown();

  return 0;
}
