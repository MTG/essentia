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

    Algorithm* audioStereo = factory.create("AudioLoader",
                                    "filename", audioFilename);

    Algorithm* monoMixer = factory.create("MonoMixer");

    Algorithm* realAccumulator = factory.create("RealAccumulator");

    Algorithm* frameCutter = factory.create("FrameCutter",
                                            "frameSize", framesize,
                                            "hopSize", hopsize,
                                            "startFromZero", true);


    Algorithm* discontinuityDetector    = factory.create("DiscontinuityDetector",
                                                         "detectionThreshold", 15,
                                                         "frameSize", framesize, 
                                                         "hopSize", hopsize,
                                                         "silenceThreshold", -25);

    Algorithm* gapsDetector             = factory.create("GapsDetector",
                                                         "frameSize", framesize, 
                                                         "hopSize", hopsize,
                                                         "silenceThreshold", -70); // in the first itteration of the assessment it was found
                                                                                   // that low level noise was sometimes considered noise 
                                                         
    Algorithm* startStopCut             = factory.create("StartStopCut",
                                                         "maximumStartTime", 1, // Found song with only this margin (to double-check)
                                                         "maximumStopTime", 1);


    Algorithm* saturationDetector       = factory.create("SaturationDetector",
                                                         "frameSize", framesize, 
                                                         "hopSize", hopsize,
                                                         "differentialThreshold", 0.0001,
                                                         "minimumDuration", 2.0f); // An experiment on rock songs showed that distortion is evident when 
                                                                                   // the median duration of the saturated regions is around 2ms

    Algorithm* truePeakDetector         = factory.create("TruePeakDetector",
                                                         "threshold", 0.0f); 

    // The algorithm should skip beginings.
    Algorithm* clickDetector            = factory.create("ClickDetector",
                                                         "frameSize", framesize, 
                                                         "hopSize", hopsize,
                                                         "silenceThreshold", -25, // This is too high. Just a work around to the problem on initial and final non-silent parts
                                                         "detectionThreshold", 38); // Experiments showed that a higher threshold is not eenough to detect audible clicks.

    Algorithm* loudnessEBUR128          = factory.create("LoudnessEBUR128");

    Algorithm* humDetector              = factory.create("HumDetector",
                                                         "sampleRate", sr,
                                                         "minimumDuration", 20.f,  // [seconds] We are only interested in humming tones if they are present over very long segments 
                                                         "frameSize", .4,  // For this algorithm, `frameSize` `hopSoze` are expresed in seconds.
                                                         "hopSize", .2); 

    Algorithm* snr                      = factory.create("SNR",
                                                         "frameSize", framesize,
                                                         "sampleRate", sr); 

    Algorithm* startStopSilence         = factory.create("StartStopSilence"); 

    Algorithm* windowing                = factory.create("Windowing",
                                                         "size", framesize,
                                                         "zeroPadding", 0,
                                                         "type", "hann",
                                                         "normalized", false);

    Algorithm* noiseBurstDetector       = factory.create("NoiseBurstDetector", 
                                                         "threshold", 200);

    Algorithm* falseStereoDetector      = factory.create("FalseStereoDetector");

    
    cout << "-------- connecting algos ---------" << endl;


    audioStereo->output("audio")                   >>  monoMixer->input("audio");
    audioStereo->output("audio")                   >>  loudnessEBUR128->input("signal");
    audioStereo->output("audio")                   >>  falseStereoDetector->input("audio");
    audioStereo->output("numberChannels")          >>  monoMixer->input("numberChannels");
    audioStereo->output("sampleRate")              >>  NOWHERE;
    audioStereo->output("md5")                     >>  NOWHERE;
    audioStereo->output("bit_rate")                >>  NOWHERE;
    audioStereo->output("codec")                   >>  NOWHERE;

    monoMixer->output("audio")                     >>  frameCutter->input("signal");
    monoMixer->output("audio")                     >>  realAccumulator->input("data");
    monoMixer->output("audio")                     >>  humDetector->input("signal");

    realAccumulator->output("array")               >>  startStopCut->input("audio");
    realAccumulator->output("array")               >>  truePeakDetector->input("signal");

    startStopCut->output("startCut")               >>  PC(pool, "startStopCut.start");
    startStopCut->output("stopCut")                >>  PC(pool, "startStopCut.cut");

    truePeakDetector->output("peakLocations")      >>  PC(pool, "truePeakDetector.peakLocations");
    truePeakDetector->output("output")             >>  NOWHERE;

    // Time domain algorithms do not require Windowing.
    frameCutter->output("frame")                   >>  discontinuityDetector->input("frame");
    frameCutter->output("frame")                   >>  gapsDetector->input("frame");
    frameCutter->output("frame")                   >>  saturationDetector->input("frame");
    frameCutter->output("frame")                   >>  clickDetector->input("frame");
    frameCutter->output("frame")                   >>  startStopSilence->input("frame");
    frameCutter->output("frame")                   >>  noiseBurstDetector->input("frame");
    
    frameCutter->output("frame")                   >>  windowing->input("frame");
    windowing->output("frame")                     >>  snr->input("frame");


    // Store the outputs in the Pool.
    loudnessEBUR128->output("momentaryLoudness")   >>  NOWHERE;
    loudnessEBUR128->output("shortTermLoudness")   >>  NOWHERE;
    loudnessEBUR128->output("integratedLoudness")  >>  PC(pool, "loudnessEBUR128.integratedLoudness");
    loudnessEBUR128->output("loudnessRange")       >>  PC(pool, "loudnessEBUR128.loudnessRange");

    humDetector->output("r")                       >>  NOWHERE;
    humDetector->output("frequencies")             >>  PC(pool, "humDetector.frequencies");
    humDetector->output("saliences")               >>  PC(pool, "humDetector.saliences");
    humDetector->output("starts")                  >>  PC(pool, "humDetector.starts");
    humDetector->output("ends")                    >>  PC(pool, "humDetector.ends");

    falseStereoDetector->output("correlation")     >>  PC(pool, "falseStereoDetector.correlation");
    falseStereoDetector->output("isFalseStereo")   >>  NOWHERE;

    connectSingleValue(discontinuityDetector->output("discontinuityLocations"), pool, "discontinuities.locations");
    connectSingleValue(discontinuityDetector->output("discontinuityAmplitudes"), pool,  "discontinuities.amplitudes");

    connectSingleValue(gapsDetector->output("starts"), pool, "gaps.starts");
    connectSingleValue(gapsDetector->output("ends"), pool, "gaps.ends");

    connectSingleValue(saturationDetector->output("starts"), pool, "saturationDetector.starts");
    connectSingleValue(saturationDetector->output("ends"), pool, "saturationDetector.ends");

    connectSingleValue(clickDetector->output("starts"), pool, "clickDetector.starts");
    connectSingleValue(clickDetector->output("ends"), pool, "clickDetector.ends");

    snr->output("instantSNR")                     >>  NOWHERE;
    snr->output("spectralSNR")                    >>  NOWHERE;
    connectSingleValue(snr->output("averagedSNR"), pool, "snr.averagedSNR");


    connectSingleValue(startStopSilence->output("startFrame"), pool, "startStopSilence.startFrame");
    connectSingleValue(startStopSilence->output("stopFrame"), pool, "startStopSilence.stopFrame");
    
    connectSingleValue(noiseBurstDetector->output("indexes"), pool, "noiseBurstDetector.indexes");


    cout << "-------- running algos ---------" << endl;
    Network(audioStereo).run();

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
