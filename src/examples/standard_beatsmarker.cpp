#include <iostream>
#include <fstream>
#include "algorithmfactory.h"
#include "pool.h"
using namespace std;
using namespace essentia;
using namespace standard;

int main(int argc, char* argv[]) {

  if (argc != 3) {
    cout << "ERROR: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input output_file" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////
  Real sampleRate = 44100.0;

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audioLoader = factory.create("MonoLoader",
                                          "filename", audioFilename,
                                          "sampleRate", sampleRate);

  Algorithm* beatTracker = factory.create("BeatTrackerMultiFeature");


  vector<Real> audio;
  vector<Real> beats;

  audioLoader->output("audio").set(audio);
  audioLoader->compute();

  beatTracker->input("signal").set(audio);
  beatTracker->output("ticks").set(beats);
  beatTracker->compute();


  vector<Real> audioOutput;

  Algorithm* beatsMarker = factory.create("AudioOnsetsMarker",
                                          "onsets", beats,
                                          "type", "beep");

  Algorithm* audioWriter = factory.create("MonoWriter",
                                          "filename", outputFilename,
                                          "sampleRate", sampleRate);

  beatsMarker->input("signal").set(audio);
  beatsMarker->output("signal").set(audioOutput);

  audioWriter->input("audio").set(audioOutput);

  beatsMarker->compute();
  audioWriter->compute();

  delete audioLoader;
  delete beatsMarker;
  delete audioWriter;

  essentia::shutdown();

  return 0;
}
