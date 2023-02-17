
#include <iostream>
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include "credit_libav.h"
#include <essentia/utils/tnt/tnt2vector.h>

using namespace essentia;
using namespace essentia::standard;
using namespace std;
int main (int argc,char* argv[]) {

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
  int sampleRate = 44100;

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* hd    = factory.create("HumDetector",
                                    "timeWindow", 10.f,
                                    "minimumDuration", 0.1f);                                                               

  cout << "-------- connecting algos ---------" << endl;

  // Audio -> FrameCutter
  vector<Real> audioBuffer;

  audio->output("audio").set(audioBuffer);
  hd->input("signal").set(audioBuffer);

  vector<Real> a, f, s, e;
  TNT::Array2D<Real> r;

  hd->output("saliences").set(a);
  hd->output("frequencies").set(f);
  hd->output("starts").set(s);
  hd->output("ends").set(e);
  hd->output("r").set(r);



  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  audio->compute();
  hd->compute();

  pool.add("saliences", a);
  pool.add("frequencies", f);
  pool.add("starts", s);
  pool.add("ends", e);
  pool.add("r", r);

  // write results to file
  cout << "-------- writing results to file " << outputFilename << " ---------" << endl;

  Algorithm* output = AlgorithmFactory::create("YamlOutput",
                                               "filename", outputFilename);

  output->input("pool").set(pool);
  output->compute();                     

  delete audio;
  delete hd;

  essentia::shutdown();

  return 0;
}
