
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
    cout << "Usage: " << argv[0] << " input_audiofile output_jsonfile (use '-' for stdout)" << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];

  // register the algorithms in the factory(ies)
  essentia::init();

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Algorithm* audio = factory.create("AudioLoader", "filename", audioFilename);
  Algorithm* le    = factory.create("LoudnessEBUR128");                                                             
  Real sr;
  int ch, br;
  std::string md5, cod;

  vector<StereoSample> audioBuffer;
  audio->output("audio").set(audioBuffer);
  audio->output("sampleRate").set(sr);
  audio->output("numberChannels").set(ch);
  audio->output("md5").set(md5);
  audio->output("bit_rate").set(br);
  audio->output("codec").set(cod);

  le->input("signal").set(audioBuffer);

  vector<Real> momentaryLoudness, shortTermLoudness;
  Real integratedLoudness, loudnessRange;

  le->output("momentaryLoudness").set(momentaryLoudness);
  le->output("shortTermLoudness").set(shortTermLoudness);
  le->output("integratedLoudness").set(integratedLoudness);
  le->output("loudnessRange").set(loudnessRange);

  audio->compute();
  le->compute();

  Pool pool = Pool();
  pool.set("loudness_ebu128.momentary", momentaryLoudness);
  pool.set("loudness_ebu128.short_term", shortTermLoudness);
  pool.set("loudness_ebu128.integrated", integratedLoudness);
  pool.set("loudness_ebu128.loudness_range", loudnessRange);

  Algorithm* yaml_writer = factory.create("YamlOutput",
                               "filename", outputFilename, "format", "json");
  yaml_writer->input("pool").set(pool);
  yaml_writer->compute();

  delete audio;
  delete le;
  delete yaml_writer;

  essentia::shutdown();
  return 0;
}
