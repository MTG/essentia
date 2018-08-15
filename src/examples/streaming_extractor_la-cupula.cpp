/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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

#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/scheduler/network.h>
#include <essentia/streaming/algorithms/poolstorage.h>

#include "credit_libav.h"

using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


int essentia_main(string audioFilename, string outputFilename) {
  // Returns: 1 on essentia error

  try {
    essentia::init();

    cout.precision(10); // TODO ????

    // instanciate factory and create algorithms:
    streaming::AlgorithmFactory& factory = streaming::AlgorithmFactory::instance();

    Real sr = 44100.f;
    int framesize = 512;
    int hopsize = 256;

    Pool pool;

    Algorithm* audioload                = factory.create("AudioLoader",
                                                         "filename", audioFilename);

    Algorithm* monoMixer                = factory.create("MonoMixer");

    Algorithm* frameCutter              = factory.create("FrameCutter",
                                                         "frameSize", framesize,
                                                         "hopSize", hopsize,
                                                         "startFromZero", true);


    Algorithm* discontinuityDetector    = factory.create("DiscontinuityDetector",
                                                         "frameSize", framesize, 
                                                         "hopSize", hopsize);

    Algorithm* gapsDetector             = factory.create("GapsDetector",
                                                         "frameSize", framesize, 
                                                         "hopSize", hopsize); 
                                                         
    Algorithm* startStopCut             = factory.create("StartStopCut");

    Algorithm* realAccumulator          = factory.create("RealAccumulator");

    Algorithm* saturationDetector       = factory.create("SaturationDetector",
                                                         "frameSize", framesize, 
                                                         "hopSize", hopsize);

    Algorithm* truePeakDetector         = factory.create("TruePeakDetector"); 

    Algorithm* clickDetector            = factory.create("ClickDetector",
                                                         "frameSize", framesize, 
                                                         "hopSize", hopsize);

    Algorithm* humDetector              = factory.create("HumDetector"); 
    Algorithm* loudnessEBUR128          = factory.create("LoudnessEBUR128"); 
    Algorithm* snr                      = factory.create("SNR"); 
    Algorithm* startStopSilence         = factory.create("StartStopSilence"); 



     Algorithm* windowing               = factory.create("Windowing",
                                                         "type", "hann",
                                                         "zeroPadding", 0);

    
    cout << "-------- connecting algos ---------" << endl;


    audioload->output("audio")                    >>  monoMixer->input("audio");
    audioload->output("audio")                    >>  loudnessEBUR128->input("signal");
    audioload->output("sampleRate")               >>  NOWHERE;
    audioload->output("numberChannels")           >>  monoMixer->input("numberChannels");
    audioload->output("md5")                      >>  NOWHERE;
    audioload->output("bit_rate")                 >>  NOWHERE;
    audioload->output("codec")                    >>  NOWHERE;

    monoMixer->output("audio")                    >>  frameCutter->input("signal");
    monoMixer->output("audio")                    >>  realAccumulator->input("data");
    monoMixer->output("audio")                    >>  humDetector->input("signal");

    realAccumulator->output("array")              >>  startStopCut->input("audio");
    realAccumulator->output("array")              >>  truePeakDetector->input("signal");

    startStopCut->output("startCut")              >>  PC(pool, "startStopCut.start");
    startStopCut->output("stopCut")               >>  PC(pool, "startStopCut.cut");

    truePeakDetector->output("peakLocations")     >>  PC(pool, "truePeakDetector.peakLocations");
    truePeakDetector->output("output")            >>  NOWHERE;

    // Time domain algorithms do not require Windowing.
    frameCutter->output("frame")                  >>  discontinuityDetector->input("frame");
    frameCutter->output("frame")                  >>  gapsDetector->input("frame");
    frameCutter->output("frame")                  >>  saturationDetector->input("frame");
    frameCutter->output("frame")                  >>  clickDetector->input("frame");
    frameCutter->output("frame")                  >>  startStopSilence->input("frame");
    
    frameCutter->output("frame")                  >>  windowing->input("frame");
    windowing->output("frame")                    >>  snr->input("frame");


    // Store the outputs in the Pool.
    discontinuityDetector->output("discontinuityLocations")       >>  PC(pool, "discontinuities.locations");
    discontinuityDetector->output("discontinuityAmplitudes")      >>  PC(pool, "discontinuities.durations");

    gapsDetector->output("starts")                                >>  PC(pool, "gaps.starts");
    gapsDetector->output("ends")                                  >>  PC(pool, "gaps.ends");

    saturationDetector->output("starts")                          >>  PC(pool, "saturationDetector.starts");
    saturationDetector->output("ends")                            >>  PC(pool, "saturationDetector.ends");

    clickDetector->output("starts")                               >>  PC(pool, "clickDetector.starts");
    clickDetector->output("ends")                                 >>  PC(pool, "clickDetector.ends");

    loudnessEBUR128->output("momentaryLoudness")                  >>  NOWHERE;
    loudnessEBUR128->output("shortTermLoudness")                  >>  NOWHERE;
    loudnessEBUR128->output("integratedLoudness")                 >>  PC(pool, "loudnessEBUR128.integratedLoudness");
    loudnessEBUR128->output("loudnessRange")                      >>  PC(pool, "loudnessEBUR128.loudnessRange");

    humDetector->output("r")                                      >>  NOWHERE;
    humDetector->output("frequencies")                            >>  PC(pool, "humDetector.frequencies");
    humDetector->output("saliences")                              >>  PC(pool, "humDetector.saliences");
    humDetector->output("starts")                                 >>  PC(pool, "humDetector.starts");
    humDetector->output("ends")                                   >>  PC(pool, "humDetector.ends");

    snr->output("instantSNR")                                     >>  NOWHERE;
    snr->output("averagedSNR")                                    >>  PC(pool, "snr.averagedSNR");
    snr->output("spectralSNR")                                    >>  PC(pool, "snr.spectralSNR");

    startStopSilence->output("startFrame")                        >>  PC(pool, "startStopSilence.startFrame");
    startStopSilence->output("stopFrame")                         >>  PC(pool, "startStopSilence.stopFrame");



    cout << "-------- running algos ---------" << endl;
    Network(audioload).run();

    cout << "-------- writting Yalm ---------" << endl;

    // Write to yaml file
    standard::Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                                     "filename", outputFilename);
    output->input("pool").set(pool);
    output->compute();
    delete output;

    essentia::shutdown();
    cout << "-------- Done! ---------" << endl;
  }
  catch (EssentiaException& e) {
    cerr << e.what() << endl;
    return 1;
  }
  return 0;

}

int main(int argc, char* argv[]) {

  string audioFilename, outputFilename;

  switch (argc) {
    case 3:
      audioFilename =  argv[1];
      outputFilename = argv[2];
      break;
    default:
      return -1;
  }

  return essentia_main(audioFilename, outputFilename);
}
