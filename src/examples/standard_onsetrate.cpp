#include <iostream>
#include "algorithmfactory.h"
using namespace std;
using namespace essentia;
using namespace standard;

int main(int argc, char* argv[]) {

  if (argc != 2) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile" << endl;
    exit(1);
  }

  essentia::init();

  Real onsetRate;
  vector<Real> onsets;

  vector<Real> audio, unused;

  // File Input
  Algorithm* audiofile = AlgorithmFactory::create("MonoLoader",
                                                  "filename", argv[1],
                                                  "sampleRate", 44100);

  Algorithm* extractoronsetrate = AlgorithmFactory::create("OnsetRate");

  audiofile->output("audio").set(audio);

  extractoronsetrate->input("signal").set(audio);
  extractoronsetrate->output("onsets").set(onsets);
  extractoronsetrate->output("onsetRate").set(onsetRate);

  audiofile->compute();
  extractoronsetrate->compute();

  cout << "onsetRate: " << onsetRate << endl;
  cout << "onsetTimes: " << onsets << endl;

  delete extractoronsetrate;
  delete audiofile;

  essentia::shutdown();

  return 0;
}
