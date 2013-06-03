#include <iostream>
#include <fstream>
#include "algorithmfactory.h"
#include "poolstorage.h"
#include "network.h"
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
  int frameSize = 1024;
  int hopSize = 512;


  streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize);

  Algorithm* w     = factory.create("Windowing",
                                    "type", "blackmanharris62");

  Algorithm* spec  = factory.create("Spectrum");
  Algorithm* gfcc  = factory.create("GFCC","highFrequencyBound",9795, "lowFrequencyBound",26,"numberBands",18);


  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos --------" << endl;

  // Audio -> FrameCutter
  connect(audio->output("audio"), fc->input("signal"));

  // FrameCutter -> Windowing -> Spectrum
  connect(fc->output("frame"), w->input("frame"));
  connect(w->output("frame"), spec->input("frame"));

  // Spectrum -> MFCC -> Pool
  connect(spec->output("spectrum"), gfcc->input("spectrum"));
  connect(gfcc->output("bands"), pool, "lowlevel.gfcc_bands");     
  connect(gfcc->output("gfcc"), pool, "lowlevel.gfcc"); 
  //connect(gfcc->output("gfcc"), NOWHERE); 



  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  Network network(audio);
  network.run();

  // write results to file
  cout << "-------- writing results to file " << outputFilename << " --------" << endl;

  standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                  "filename", outputFilename);
  output->input("pool").set(pool);
  output->compute();

  delete output;
  essentia::shutdown();

  return 0;
}
