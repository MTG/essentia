#include <iostream>
#include "algorithmfactory.h"
#include "network.h"
#include "../scheduler/network.h"
#include "poolstorage.h"
using namespace std;
using namespace essentia;
using namespace streaming;
using namespace scheduler;

int main(int argc, char* argv[]) {

  if (argc != 3) {
    cout << "ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input yaml_output" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////
  Real sampleRate = 44100.0;
  /*
  int frameSize = 2048;
  int hopSize = 1024;
  */

  // we want to compute the MFCC of a file: we need the create the following:
  // audioloader -> framecutter -> windowing -> FFT -> MFCC -> PoolStorage
  // we also need a DevNull which is able to gobble data without doing anything
  // with it (required otherwise a buffer would be filled and blocking)

  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("EqloudLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* tonal = factory.create("KeyExtractor");


  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  audio->output("audio")  >>  tonal->input("audio");

  connect(tonal->output("key"),      pool, "key");
  connect(tonal->output("scale"),    pool, "scale");
  connect(tonal->output("strength"), pool, "strength");


  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  Network n(audio);
  n.run();

  //runGenerator(audio);

  // write results to file
  cout << "-------- writing results to file " << outputFilename << " --------" << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                   "filename", outputFilename);
  output->input("pool").set(pool);
  output->compute();

  //deleteNetwork(audio);
  delete output;
  essentia::shutdown();

  return 0;
}
