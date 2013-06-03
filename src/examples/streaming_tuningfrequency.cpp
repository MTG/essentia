#include <iostream>
#include "algorithmfactory.h"
#include "../scheduler/network.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;

int main(int argc, char* argv[]) {

  if (argc != 2) {
    cout << "ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input" << endl;
    exit(1);
  }

  string audioFilename = argv[1];

  //setDebugLevel(EAll);
  //unsetDebugLevel(EExecution);
  //unsetDebugLevel(EMemory);

  // register the algorithms in the factory(ies)
  essentia::init();
  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename);

  Algorithm* tf    = factory.create("TuningFrequencyExtractor");


  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  try {

    audio->output("audio")         >>  tf->input("signal");
    tf->output("tuningFrequency")  >>  NOWHERE;


  /////////// STARTING THE ALGORITHMS //////////////////
    cout << "-------- start processing " << audioFilename << " --------" << endl;

    Network n(audio);
    n.run();
  }
  catch (EssentiaException& e) {
      cout << "EXC: " << e.what() << endl;
  }

  essentia::shutdown();

  return 0;
}
