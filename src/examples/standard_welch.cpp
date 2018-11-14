
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

    if (argc != 2) {
    cout << "ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input yaml_output" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  // string outputFilename = argv[2];

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////
  int sampleRate = 44100;
  int frameSize = 256;
  int hopSize = 256;

  AlgorithmFactory& factory = AlgorithmFactory::instance();


  Algorithm* audio = factory.create("MonoLoader",
                                    "filename", audioFilename,
                                    "sampleRate", sampleRate);

  Algorithm* fc    = factory.create("FrameCutter",
                                    "frameSize", frameSize,
                                    "hopSize", hopSize,
                                    "startFromZero", true);

  Algorithm* snr    = factory.create("Welch",
                                    "fftSize", hopSize,
                                    "frameSize", frameSize,
                                    "sampleRate", 1.0,
                                    "scaling","density");                                                               

  cout << "-------- connecting algos ---------" << endl;

  // Audio -> FrameCutter
  vector<Real> audioBuffer;

  audio->output("audio").set(audioBuffer);
  fc->input("signal").set(audioBuffer);

  // FrameCutter -> Windowing -> Spectrum
  vector<Real> frame, psd;
  Real AverageSNR, AverageSNRMA;
  fc->output("frame").set(frame);
  snr->input("frame").set(frame);

  snr->output("PSD").set(psd);



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

    snr->compute();

    pool.add("AverageSNR", AverageSNR);
    pool.add("AverageSNRMA", AverageSNRMA);

  }

  // write results to file
  cout << "-------- writing results to file " << "results.yaml" << " ---------" << endl;

  Algorithm* output = AlgorithmFactory::create("YamlOutput",
                                               "filename", "results.yaml");

  output->input("pool").set(pool);
  output->compute();                     

  delete audio;
  delete fc;
  delete snr;

  essentia::shutdown();

  return 0;

}