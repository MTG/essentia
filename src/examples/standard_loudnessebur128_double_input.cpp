
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

  // This examples processes 2 files with the LoudnessEBUR128.
  // It was created to debug the behavior of the algorithm 
  // on the second compute call.This algorithm does not return
  // or print any value.
  if (argc != 3) {
    cout << "ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << "audio_input_1 audio_input_2" << endl;
    exit(1);
  }

  setDebugLevel(EAll);
  string audioFilename1 = argv[1];
  string audioFilename2 = argv[2];

  // Register the algorithms in the factory(ies).
  essentia::init();


  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audio1 = factory.create("AudioLoader",
                                    "filename", audioFilename1);

  Algorithm* audio2 = factory.create("AudioLoader",
                                    "filename", audioFilename2);

  Algorithm* le     = factory.create("LoudnessEBUR128");                                                             

  cout << "-------- connecting algos ---------" << endl;
  Real sr;
  int ch, br;
  std::string md5, cod;

  // Audio -> FrameCutter
  vector<StereoSample> audioBuffer;
  audio1->output("audio").set(audioBuffer);
  audio1->output("sampleRate").set(sr);
  audio1->output("numberChannels").set(ch);
  audio1->output("md5").set(md5);
  audio1->output("bit_rate").set(br);
  audio1->output("codec").set(cod);

  le->input("signal").set(audioBuffer);

  // FrameCutter -> GapsDetector
  vector<Real> momentaryLoudness, shortTermLoudness;
  Real integratedLoudness, loudnessRange;

  le->output("momentaryLoudness").set(momentaryLoudness);
  le->output("shortTermLoudness").set(shortTermLoudness);
  le->output("integratedLoudness").set(integratedLoudness);
  le->output("loudnessRange").set(loudnessRange);


  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename1 << " --------" << endl;
  audio1->compute();
  le->compute();

  // audioBuffer.clear();
  audio2->output("audio").set(audioBuffer); 
  audio2->output("audio").set(audioBuffer);
  audio2->output("sampleRate").set(sr);
  audio2->output("numberChannels").set(ch);
  audio2->output("md5").set(md5);
  audio2->output("bit_rate").set(br);
  audio2->output("codec").set(cod);

  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename2 << " --------" << endl;
  audio2->compute();
  le->compute();
                    
  delete audio1;
  delete audio2;
  delete le;

  essentia::shutdown();

  return 0;
}
