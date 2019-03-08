
#include <iostream>
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include "credit_libav.h"


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
  int frameSize = 512;
  int hopSize = 256;

  AlgorithmFactory& factory = AlgorithmFactory::instance();


  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize,
                                    "startFromZero", true);

  Algorithm* dd    = factory.create("DiscontinuityDetector");                                                               

  cout << "-------- connecting algos ---------" << endl;

  vector<Real> audioBuffer;

  audio->output("audio").set(audioBuffer);
  fc->input("signal").set(audioBuffer);

  vector<Real> frame, peaks, amps;

  fc->output("frame").set(frame);
  dd->input("frame").set(frame);

  dd->output("discontinuityLocations").set(peaks);
  dd->output("discontinuityAmplitudes").set(amps);


  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  audio->compute();

  while (true) {

    // compute a frame
    fc->compute();

    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }

    // if the frame is silent, just drop it and go on processing
    // if (isSilent(frame)) continue;

    dd->compute();

    pool.add("peaks", peaks);
    pool.add("amps", amps);
  }

  // write results to file
  cout << "-------- writing results to file " << outputFilename << " ---------" << endl;

  Algorithm* output = AlgorithmFactory::create("YamlOutput",
                                               "filename", outputFilename);

  output->input("pool").set(pool);
  output->compute();                     

  delete audio;
  delete fc;
  delete dd;

  essentia::shutdown();

  return 0;
}
