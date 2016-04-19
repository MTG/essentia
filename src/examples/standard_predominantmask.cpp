  /*
 * Copyright (C) 2006-2015  Music Technology Group - Universitat Pompeu Fabra
 *
 * This file is part of Essentia
 *
 * Essentia is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Affero General Public License as published by the Free
 * Software Foundation (FSF), either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the Affero GNU General Public License
 * version 3 along with this program.  If not, see http://www.gnu.org/licenses/
 */

#include <iostream>
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/pool.h>
using namespace std;
using namespace essentia;
using namespace standard;


void scaleAudioVector(vector<Real> &buffer, const Real scale)
{
for (int i=0; i < int(buffer.size()); ++i){
    buffer[i] = scale * buffer[i];
}
}

int main(int argc, char* argv[]) {

  if (argc < 3) {
    cout << "Error: incorrect number of arguments." << endl;
    cout << "Usage: " << argv[0] << " audio_input output_file [attenuation_dB]" << endl;
    cout << "attenuation_dB (optional): value in dB's of the attenuation applied to the predominant pitched component. \n \
    A positive value 'mutes' the pitched component. A negative value 'soloes' the pitched component." << endl;
    exit(1);
  }

  string audioFilename = argv[1];
  string outputFilename = argv[2];
  Real attenuation_dB = -200.f;
  if (argc ==4){
   string attstr = argv[3];
   sscanf(attstr.c_str(), "%f",&attenuation_dB);
  }

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  /////// PARAMS //////////////

  /////// PARAMS //////////////
  int framesize = 2048;
  int hopsize = 128; // 128 for predominant melody
  Real sr = 44100;
  bool usePredominant = true; // set to true if PredmonantMelody extraction is used. Set to false if monhonic Pitch-YinFFT is used,



  AlgorithmFactory& factory = AlgorithmFactory::instance();

    Algorithm* audioLoader    = factory.create("MonoLoader",
                                           "filename", audioFilename,
                                           "sampleRate", sr,
                                           "downmix", "mix");

  Algorithm* frameCutter  = factory.create("FrameCutter",
                                           "frameSize", framesize,
                                           "hopSize", hopsize,
                                         //  "silentFrames", "noise",
                                           "startFromZero", false );

  Algorithm* window       = factory.create("Windowing", "type", "hann");

  Algorithm* equalLoudness = factory.create("EqualLoudness");

  Algorithm* fft     = factory.create("FFT",
                            "size", framesize);

  Algorithm* predominantMelody = factory.create("PredominantPitchMelodia", //PredominantMelody",
                                                "frameSize", framesize,
                                                "hopSize", hopsize,
                                                "sampleRate", sr);
//
  Algorithm* spectrum = factory.create("Spectrum",
                                       "size", framesize);

  Algorithm* pitchDetect = factory.create("PitchYinFFT",
                                          "frameSize", framesize,
                                          "sampleRate", sr);

//  Algorithm* realAccumulator     = factory.create("RealAccumulator");


  Algorithm* harmonicMask     = factory.create("HarmonicMask",
                            "sampleRate", sr,
                            "binWidth", 2,
                            "attenuation", attenuation_dB);


  Algorithm* ifft     = factory.create("IFFT",
                                "size", framesize);

  Algorithm* overlapAdd = factory.create("OverlapAdd",
                                            "frameSize", framesize,
                                           "hopSize", hopsize);


  Algorithm* audioWriter = factory.create("MonoWriter",
                                     "filename", outputFilename);



  vector<Real> pitchIn;
  vector<Real> pitchConf;
  vector<Real> audio;
  vector<Real> eqaudio;
  vector<Real> frame;
  vector<Real> wframe;
  vector<complex<Real> > fftframe;
  vector<complex<Real> > fftmaskframe;
  vector<Real> ifftframe;
  vector<Real> alladuio; // concatenated audio file output
 // Real confidence;

  // analysis
  audioLoader->output("audio").set(audio);

  equalLoudness->input("signal").set(audio);
  equalLoudness->output("signal").set(eqaudio);

  predominantMelody->input("signal").set(eqaudio);
  predominantMelody->output("pitch").set(pitchIn);
  predominantMelody->output("pitchConfidence").set(pitchConf);

  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

  window->input("frame").set(frame);
  window->output("frame").set(wframe);

  // set spectrum:
  vector<Real> spec;
  spectrum->input("frame").set(wframe);
  spectrum->output("spectrum").set(spec);

  // set Yin pitch extraction:
  Real thisPitch = 0., thisConf = 0;
  pitchDetect->input("spectrum").set(spec);
  pitchDetect->output("pitch").set(thisPitch);
  pitchDetect->output("pitchConfidence").set(thisConf);


  fft->input("frame").set(wframe);
  fft->output("fft").set(fftframe);

  // processing harmonic mask (apply mask)
  harmonicMask->input("fft").set(fftframe);
  harmonicMask->input("pitch").set(thisPitch);
  harmonicMask->output("fft").set(fftmaskframe);

  // Synthesis
  ifft->input("fft").set(fftmaskframe);
  ifft->output("frame").set(ifftframe);


  vector<Real> audioOutput;

  overlapAdd->input("signal").set(ifftframe); // or frame ?
  overlapAdd->output("signal").set(audioOutput);




////////
/////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing " << audioFilename << " --------" << endl;

  audioLoader->compute();
  equalLoudness->compute();
  predominantMelody->compute();

  int counter = 0;

  while (true) {

    // compute a frame
    frameCutter->compute();

    // if it was the last one (ie: it was empty), then we're done.
    if (!frame.size()) {
      break;
    }

    window->compute();

    // pitch extraction
    spectrum->compute();
    pitchDetect->compute();

    // get predominant pitch
    if (usePredominant){
      thisPitch = pitchIn[counter];
    }


    fft->compute();
    harmonicMask-> compute();
    ifft->compute();
    overlapAdd->compute();

    counter++;

    alladuio.insert(alladuio.end(), audioOutput.begin(), audioOutput.end());

  }



  // write results to file
  cout << "-------- writing results to file " << outputFilename << " ---------" << endl;

    // write to output file
    audioWriter->input("audio").set(alladuio);
    audioWriter->compute();


  delete audioLoader;
  delete frameCutter;
  delete fft;
  delete predominantMelody;
  delete pitchDetect;
  //delete realAccumulator;
  delete harmonicMask;
  delete ifft;
  delete overlapAdd;
  delete audioWriter;

  essentia::shutdown();

  return 0;
}


