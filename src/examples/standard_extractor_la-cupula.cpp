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
using namespace essentia::standard;


int essentia_main(string audioFilename, string outputFilename) {
  // Returns: 1 on essentia error

  try {
    essentia::init();

    cout.precision(10); // TODO ????

    // instanciate factory and create algorithms:
    AlgorithmFactory& factory = AlgorithmFactory::instance();

    Real sr = 44100.f;
    int framesize = 512;
    int hopsize = 256;
    int silenceThreshold = -25;

    Pool pool;

    // Algorithm instantiations
    Algorithm* audioStereo = factory.create("AudioLoader",
                                    "filename", audioFilename);

    Algorithm* monoMixer = factory.create("MonoMixer");

    Algorithm* frameCutter = factory.create("FrameCutter",
                                            "frameSize", framesize,
                                            "hopSize", hopsize,
                                            "startFromZero", true);


    Algorithm* discontinuityDetector    = factory.create("DiscontinuityDetector",
                                                         "detectionThreshold", 15,
                                                         "frameSize", framesize, 
                                                         "hopSize", hopsize,
                                                         "silenceThreshold", silenceThreshold);

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
                                                         "threshold", 0.0f,
                                                         "quality", 2); // Reducing the quality of the conversion doubles the conversion speed.   

    // The algorithm should skip beginings.
    Algorithm* clickDetector            = factory.create("ClickDetector",
                                                         "frameSize", framesize, 
                                                         "hopSize", hopsize,
                                                         "silenceThreshold", silenceThreshold, // This is too high. Just a work around to the problem on initial and final non-silent parts
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
                                                         "threshold", 200,
                                                         "silenceThreshold", silenceThreshold);

    Algorithm* falseStereoDetector      = factory.create("FalseStereoDetector");


    cout << "-------- connecting algos ---------" << endl;
    Real fs;
    int ch, br;
    std::string md5, cod;

    vector<StereoSample> audioBuffer;
    audioStereo->output("audio").set(audioBuffer);
    audioStereo->output("sampleRate").set(fs);
    audioStereo->output("numberChannels").set(ch);
    audioStereo->output("md5").set(md5);
    audioStereo->output("bit_rate").set(br);
    audioStereo->output("codec").set(cod);


    vector<Real> momentaryLoudness, shortTermLoudness;
    Real integratedLoudness, loudnessRange;
    loudnessEBUR128->input("signal").set(audioBuffer);
    loudnessEBUR128->output("momentaryLoudness").set(momentaryLoudness);
    loudnessEBUR128->output("shortTermLoudness").set(shortTermLoudness);
    loudnessEBUR128->output("integratedLoudness").set(integratedLoudness);
    loudnessEBUR128->output("loudnessRange").set(loudnessRange);


    Real correlation;
    int isFalseStereo;
    falseStereoDetector->input("frame").set(audioBuffer);
    falseStereoDetector->output("isFalseStereo").set(isFalseStereo);
    falseStereoDetector->output("correlation").set(correlation);   


    vector<Real> audio;
    monoMixer->input("audio").set(audioBuffer);
    monoMixer->input("numberChannels").set(2);
    monoMixer->output("audio").set(audio);


    int startStopCutStart, startStopCutEnd; 
    startStopCut->input("audio").set(audio);
    startStopCut->output("startCut").set(startStopCutStart);
    startStopCut->output("stopCut").set(startStopCutEnd);


    TNT::Array2D<Real> r;
    vector<Real> humFrequencies, humSaliences, humStarts, humEnds;
    humDetector->input("signal").set(audio);
    humDetector->output("r").set(r);
    humDetector->output("frequencies").set(humFrequencies);
    humDetector->output("saliences").set(humSaliences);
    humDetector->output("starts").set(humStarts);
    humDetector->output("ends").set(humEnds);


    std::vector<Real> peakLocations, truePeakDetectorOutput;
    truePeakDetector->input("signal").set(audio);
    truePeakDetector->output("peakLocations").set(peakLocations);
    truePeakDetector->output("output") .set(truePeakDetectorOutput);


    std::vector<Real> frame;
    frameCutter->input("signal").set(audio);
    frameCutter->output("frame").set(frame);


    // Time domain algorithms do not require Windowing.
    std::vector<Real> discontinuityLocations, discontinuityAmplitudes;
    discontinuityDetector->input("frame").set(frame);
    discontinuityDetector->output("discontinuityLocations").set(discontinuityLocations);
    discontinuityDetector->output("discontinuityAmplitudes").set(discontinuityAmplitudes);


    std::vector<Real> gapsDetectorStarts, gapsDetectorEnds;
    gapsDetector->input("frame").set(frame);
    gapsDetector->output("starts").set(gapsDetectorStarts);
    gapsDetector->output("ends").set(gapsDetectorEnds);


    std::vector<Real> saturationDetectorStarts, saturationDetectorEnds;
    saturationDetector->input("frame").set(frame);
    saturationDetector->output("starts").set(saturationDetectorStarts);
    saturationDetector->output("ends").set(saturationDetectorEnds);


    std::vector<Real> clickDetectorStarts, clickDetectorEnds;
    clickDetector->input("frame").set(frame);
    clickDetector->output("starts").set(clickDetectorStarts);
    clickDetector->output("ends").set(clickDetectorEnds);


    int startFrame, stopFrame;
    startStopSilence->input("frame").set(frame);
    startStopSilence->output("startFrame").set(startFrame);
    startStopSilence->output("stopFrame").set(stopFrame);

    vector<Real> noiseBurstIndexes;
    noiseBurstDetector->input("frame").set(frame);
    noiseBurstDetector->output("indexes").set(noiseBurstIndexes);

    std::vector<Real> windowedFrame;
    windowing->input("frame").set(frame);
    windowing->output("frame").set(windowedFrame);


    Real averagedSNR, instantSNR;
    std::vector<Real> spectralSNR;
    snr->input("frame").set(windowedFrame);
    snr->output("instantSNR").set(instantSNR);
    snr->output("averagedSNR").set(averagedSNR);
    snr->output("spectralSNR").set(spectralSNR);


    cout << "-------- running algos ---------" << endl;
    audioStereo->compute();

    pool.set("filename", audioFilename);

    pool.set("duration", audioBuffer.size() / sr);

    loudnessEBUR128->compute();
    pool.set("EBUR128.integratedLoudness", integratedLoudness);
    pool.set("EBUR128.range", loudnessRange);

    falseStereoDetector->compute();
    pool.set("channelsCorrelation", correlation);

    monoMixer->compute();

    startStopCut->compute();
    pool.set("startStopCut.start", startStopCutStart);
    pool.set("startStopCut.end", startStopCutEnd);

    humDetector->compute();
    if (humFrequencies.size() > 0) {
      pool.set("humDetector.present", true);
      pool.set("humDetector.frequencies", humFrequencies);
      pool.set("humDetector.saliences", humSaliences);
      pool.set("humDetector.starts", humStarts);
      pool.set("humDetector.ends", humEnds);
    } else {
      pool.set("humDetector.present", false);
    }

    truePeakDetector->compute();
    for (uint i = 0; i < peakLocations.size(); i++) 
      peakLocations[i] /= sr;

    if (peakLocations.size() > 0) {
      pool.set("truePeakDetector.present", true);
      pool.set("truePeakDetector.locations", peakLocations);
    } else {
      pool.set("truePeakDetector.present", false);
    }

    vector<Real> noiseBursts;

    size_t idx = 0;
    while (true) {

      // compute a frame
      frameCutter->compute();

      // if it was the last one (ie: it was empty), then we're done.
      if (!frame.size()) {
        break;
      }

      // skippint silent frames should be internally done by each algorithm
      // if (isSilent(frame)) continue;

      discontinuityDetector->compute();

      gapsDetector->compute();

      saturationDetector->compute();

      clickDetector->compute();

      noiseBurstDetector->compute();

      windowing->compute();

      snr->compute();

      startStopSilence->compute();

      if (noiseBurstIndexes.size() > 3) {
        noiseBursts.push_back(idx * hopsize / (Real)fs);
      }

      noiseBurstIndexes.clear();

      idx++;
    }

    if (noiseBursts.size() > 0) {
      pool.set("noiseBursts.present", true);
      pool.set("noiseBursts.locations", noiseBursts);
    } else {
    pool.set("noiseBursts.present", false);
    }

    if (discontinuityLocations.size() > 0) {
      pool.set("discontinuities.present", true);
      for (uint i = 0; i < discontinuityLocations.size(); i++) 
        discontinuityLocations[i] /= sr;
      pool.set("discontinuities.locations", discontinuityLocations);
      pool.set("discontinuities.amplitudes", discontinuityAmplitudes);
    } else {
      pool.set("discontinuities.present", false);
    }

    if (gapsDetectorStarts.size() > 0) {
      pool.set("gaps.present", true);
      pool.set("gaps.starts", gapsDetectorStarts);
      pool.set("gaps.ends", gapsDetectorEnds);
    } else {
      pool.set("gaps.present", false);
    }

    if (saturationDetectorStarts.size() > 0) {
      pool.set("saturationDetector.present", true);    
      pool.set("saturationDetector.starts", saturationDetectorStarts);
      pool.set("saturationDetector.ends", saturationDetectorEnds);
    } else {
      pool.set("saturationDetector.present", false); 
    }

    if (clickDetectorStarts.size() > 0) {
      pool.set("clickDetector.present", true); 
      pool.set("clickDetector.starts", clickDetectorStarts);
      pool.set("clickDetector.ends", clickDetectorEnds);
    } else {
      pool.set("clickDetector.present", false); 
    }

    // Spectral SNR is not very relevant. 
    // pool.set("snr.spectralSNR", spectralSNR);
    pool.set("snr.averagedSNR", averagedSNR);

    pool.set("startStopSilence.start", startFrame * hopsize / fs);
    pool.set("startStopSilence.end", stopFrame * hopsize / fs);


    cout << "-------- writting Yaml ---------" << endl;
    // Write to yaml file.
    Algorithm* output = standard::AlgorithmFactory::create("YamlOutput",
                                                          "filename", outputFilename);
    output->input("pool").set(pool);
    output->compute();
    
    delete output;
    delete frameCutter;
    delete discontinuityDetector;
    delete gapsDetector;
    delete startStopCut;
    delete saturationDetector;
    delete truePeakDetector;
    delete clickDetector;
    delete windowing;
    delete humDetector;
    delete snr;
    delete loudnessEBUR128;
    delete startStopSilence;
    
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
