/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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
#include <complex>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/utils/tnt/tnt_array2d.h>

using namespace std;
using namespace essentia;
using namespace standard;

inline Real lagToBpm(Real lag, Real sampleRate, Real hopSize) {
  return 60.*sampleRate/lag/hopSize;
}

int main(int argc, char* argv[]) {

  cout << "An outdated rhythm extractor (beats, BPM, positions of tempo changes) based on TempoTap algorithm" << endl;
  cout << "Note, that using streaming_rhythmextractor is recommended instead" << endl;

  if (argc != 2) {
    cout << "Error: wrong number of arguments" << endl;
    cout << "Usage: " << argv[0] << " input_audiofile" << endl;
    exit(1);
  }

  essentia::init();

  // params:
  int framesize = 1024;
  int hopsize = 256;
  int zeropadding = 0;
  int sr = 44100;
  Real periodTolerance = 5.;
  Real bandsFreq[] = {40.0, 413.16, 974.51, 1818.94, 3089.19, 5000.0, 7874.4, 12198.29, 17181.13};
  Real bandsGain[] = {2.0, 3.0, 2.0, 1.0, 1.2, 2.0, 3.0, 2.5};
  bool useOnset = true;
  bool useBands = true;
  int frameNumber = 1024; // feature frames to buffer on
  int frameHop = 1024;    // feature frames separating 2 evaluations
  Real tolerance = 0.24;   // minimum interval between consecutive beats
  std::vector<Real> tempoHints;   // list of initial beat locations, to favour detection
                                  // of pre-determined tempo period and beats alignment
  int maxTempo = 208;     // maximum tempo allowed
  int minTempo = 40;      // minimum tempo allowed
  Real lastBeatInterval = 0.1;    // time between last beat and EOF


  // ===================== Declaring algorithms ================================

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  // File Input
  Algorithm* audioloader = factory.create("MonoLoader",
		                                      "filename", argv[1],
										                      "sampleRate", sr);
  // slice audio into frames:
  Algorithm* frameCutter = factory.create("FrameCutter",
                                           "frameSize", framesize,
                                           "hopSize", hopsize);
  // windowing algorithm:
  Algorithm* window = factory.create("Windowing",
                                     "zeroPadding", zeropadding,
                                     "size", framesize);
  // fft algorithm:
  Algorithm* fft = factory.create("FFT",
                                  "size", framesize);
  // cartesian to polar conversion:
  Algorithm* cart2polar = factory.create("CartesianToPolar");

  // onset detection: high frequency content and complex-domain
  Algorithm* onsetHfc = factory.create("OnsetDetection",
                                       "method", "hfc",
                                       "sampleRate", sr);

  Algorithm* onsetComplex = factory.create("OnsetDetection",
                                           "method", "complex",
                                           "sampleRate", sr);

  Algorithm* spectrum = factory.create("Spectrum", "size", framesize);

  // frequency bands:
  Algorithm* tempoTapBands = factory.create("FrequencyBands",
                                            "frequencyBands", arrayToVector<Real>(bandsFreq));

  // tempo scale bands
  Algorithm* tempoScaleBands = factory.create("TempoScaleBands",
                                              "bandsGain", arrayToVector<Real>(bandsGain));

  // tempo tap:
  Algorithm* tempoTap = factory.create("TempoTap",
                                       "sampleRate", sr,
                                       "numberFrames", frameNumber,
                                       "frameHop", frameHop,
                                       "frameSize", hopsize,
                                       "tempoHints", tempoHints,
                                       "minTempo", minTempo,
                                       "maxTempo", maxTempo);

  // tempo tap ticks:
  Algorithm* tempoTapTicks = factory.create("TempoTapTicks",
                                            "hopSize", hopsize,
                                            "frameHop", frameHop,
                                            "sampleRate", sr);

  // FIXME we need better rubato estimation algorithm
  // bpm rubato:
  //Algorithm* bpmRubato = factory.create("BpmRubato");


  // ====================== Setting up algorithms ==========================

  //set audio:
  vector<Real> audio;
  audioloader->output("audio").set(audio);

  // set frame creator
  vector<Real> frame;
  frameCutter->input("signal").set(audio);
  frameCutter->output("frame").set(frame);

  //set windowing:
  vector<Real> wFrame;
  window->input("frame").set(frame);
  window->output("frame").set(wFrame);

  // set fft
  vector<complex<Real> > fftFrame;
  fft->input("frame").set(wFrame);
  fft->output("fft").set(fftFrame);

  // set conversion from cartesian to polar:
  vector<Real> mag, ph;
  cart2polar->input("complex").set(fftFrame);
  cart2polar->output("magnitude").set(mag);
  cart2polar->output("phase").set(ph);

  // set onset detection:
  Real hfc;
  onsetHfc->input("spectrum").set(mag);
  onsetHfc->input("phase").set(ph);
  onsetHfc->output("onsetDetection").set(hfc);

  Real complexdomain;
  onsetComplex->input("spectrum").set(mag);
  onsetComplex->input("phase").set(ph);
  onsetComplex->output("onsetDetection").set(complexdomain);

  // for useBands = True && useOnset = False, faster than FFT+cart2polar
  spectrum->input("frame").set(wFrame);
  spectrum->output("spectrum").set(mag);

  // set frequency bands:
  vector<Real> bands;
  tempoTapBands->input("spectrum").set(mag);
  tempoTapBands->output("bands").set(bands);

  // set scaled bands:
  vector<Real> scaledBands;
  Real cumulBands;
  tempoScaleBands->input("bands").set(bands);
  tempoScaleBands->output("scaledBands").set(scaledBands);
  tempoScaleBands->output("cumulativeBands").set(cumulBands);

  // set tempotap algo
  vector<Real> features;
  vector<Real> periods, matchingPeriods;
  vector<Real> phases;
  TNT::Array2D<Real> acf;
  TNT::Array2D<Real> mcomb;

  tempoTap->input("featuresFrame").set(features);
  tempoTap->output("periods").set(periods);
  tempoTap->output("phases").set(phases);

  // set tempo tap ticks
  vector<Real> these_ticks;
  tempoTapTicks->input("periods").set(periods);
  tempoTapTicks->input("phases").set(phases);
  tempoTapTicks->output("ticks").set(these_ticks);
  tempoTapTicks->output("matchingPeriods").set(matchingPeriods);

  // vars to hold tempotap extraction
  Real bpm;                  // estimated bpm (in beats per minute)
  vector<Real> ticks;        // estimated tick locations (in seconds)
  vector<Real> estimates;    // estimated bpm per frame (in beats per minute)
  //vector<Real> rubatoStart;  // list of start times of rubato regions
  //vector<Real> rubatoStop;   // list of stop times of rubato regions
  //int rubatoNumber;          // number of rubato regions
  vector<Real> bpmIntervals; // list of beats interval (in seconds)


  // ====================== Processing ==========================

  audioloader->compute();

  int nframes = 0;
  Real oldHfc = 0.0;
  vector<Real> bpmEstimateList, periodEstimateList;

  Real fileLength = audio.size() / Real(sr);
  int startSilence = 0;
  int oldSilence = 0;
  int endSilence = int(round(fileLength * Real(sr) / hopsize) + 1);
  while (true) {

    frameCutter->compute();
    if (!frame.size()) {
       break;
    }

    window->compute();

    features.clear();

    // compute spectrum, and optionally phase vector if needed (if useOnset)
    if (useOnset) {
      fft->compute();
      cart2polar->compute();
    }
    else { // we only need the magnitude vector
      spectrum->compute();
    }

    // using onsets
    if (useOnset) {
      onsetHfc->compute();
      onsetComplex->compute();
      Real diffHfc = max(hfc - oldHfc, (Real)0.0);
      oldHfc = hfc;
      // features += ...
      features.push_back(hfc);
      features.push_back(diffHfc);
      features.push_back(complexdomain);
    }

    // using band energies
    if (useBands) {
      tempoTapBands->compute();
      tempoScaleBands->compute();
      // features += ...
      for (uint i=0; i<scaledBands.size(); i++) {
        features.push_back(scaledBands[i]);
      }
    }

    // run tempoTap and tempoTapTicks
    tempoTap->compute();
    tempoTapTicks->compute();

    // cumulate ticks
    for (uint i=0; i < these_ticks.size(); i++) {
      ticks.push_back(these_ticks[i]);
    }

    // cumulate matchingPeriods
    for (uint i=0; i < matchingPeriods.size(); i++) {
      if (matchingPeriods[i] != 0) {
        periodEstimateList.push_back(matchingPeriods[i]);
      }
    }

    // get startSilence
    if (nframes < (5.0 * sr / hopsize)) {
      if (isSilent(frame) && (startSilence == nframes - 1)) {
        startSilence = nframes;
      }
    }

    // get endSilence
    if (nframes > (fileLength - 5.0) * sr / hopsize) {
      if (isSilent(frame)) {
        /* if previous frame was not silence, have current one = endSilence */
        if (oldSilence != nframes - 1) {
          endSilence = nframes;
        }
        oldSilence = nframes;
      } /* or else, nothing to do */
    }

    nframes++;
  }

  /* make sure we do not kill beat too close to music */
  if (startSilence > 0) { startSilence --; }
  endSilence ++;

  // if the file was too short, fill the rest of the buffer with zeros
  std::fill(features.begin(),features.end(),0.f);
  // compute the beat candidates on the last incomplete buffer
  while (nframes % frameHop != 0) {
    tempoTap->compute();
    tempoTapTicks->compute();
    for (uint i=0; i < these_ticks.size(); i++) {
      ticks.push_back(these_ticks[i]);
    }
    nframes++;
  }

  // compute the last ticks and prune the ones found after the end of file
  if (ticks.size() > 2) {
    /* fill up to end of file */
    if (fileLength > ticks[ticks.size()-1]) {
      Real lastPeriod = ticks[ticks.size()-1] - ticks[ticks.size()-2];
      while (ticks[ticks.size()-1] + lastPeriod < fileLength - lastBeatInterval) {
        if (ticks[ticks.size()-1] > fileLength - lastBeatInterval) {
          break;
        }
        ticks.push_back(ticks[ticks.size()-1] + lastPeriod);
      };
    }
  }

  if (ticks.size() > 1) {
    for (int i = 0; i < (int)ticks.size(); i++) {
      /* remove all negative ticks */
      if (ticks[i] < startSilence / Real(sr) * hopsize) {
        ticks.erase(ticks.begin() + i);
        i --;
      }
    }
    for (int i = 0; i < (int)ticks.size(); i++) {
      /* kill all ticks from 350ms before the end of the song */
      if (ticks[i] > endSilence / Real(sr) * hopsize - lastBeatInterval) {
        ticks.erase(ticks.begin() + i);
        i --;
      }
    }

    for (uint i = 1; i < ticks.size(); i++) {
      /* prune all beats closer than tolerance */
      if (ticks[i] - ticks[i-1] < tolerance) {
        ticks.erase(ticks.begin() + i);
        i--;
      }
    }

    /* prune all beats doing a backward off beat */
    for (uint i = 3; i < ticks.size(); i++) {
      if ((std::abs ((ticks[i] - ticks[i-2]) - 1.5*(ticks[i] - ticks[i-1])) < 0.100) &&
        (std::abs(ticks[i] - ticks[i-1] - ticks[i-2] + ticks[i-3]) < 0.100)) {
        ticks.erase(ticks.begin() + i - 2);
        i--;
      }
    }
  }

  for (uint i=0; i< matchingPeriods.size(); i++) {
    if (matchingPeriods[i] != 0) {
      periodEstimateList.push_back(matchingPeriods[i]);
    }
  }

  for (uint i=0; i<periodEstimateList.size(); i++) {
    if (periodEstimateList[i] != 0) {
      bpmEstimateList.push_back(lagToBpm(periodEstimateList[i], Real(sr), hopsize));
    }
    else {
      bpmEstimateList.push_back(0.);
    }
  }

  if (bpmEstimateList.size() > 0) {
    Real closestBpm = 0;
    std::vector<Real> countedBins;
    for (uint i = 0; i < bpmEstimateList.size(); i++) {
      bpmEstimateList[i] /= 2.;
    }
    bincount(bpmEstimateList,countedBins);
    closestBpm = (argmax(countedBins))*2.;
    for (uint i = 0; i < bpmEstimateList.size(); i++) {
      bpmEstimateList[i] *= 2.;
      if (abs(closestBpm - bpmEstimateList[i]) < periodTolerance) {
        estimates.push_back(bpmEstimateList[i]);
      }
    }
    if (estimates.size() < 1) {
      // something odd happened
      bpm = closestBpm;
    }
    else {
      bpm = mean(estimates);
    }
  }
  else {
    bpm = 0.;
  }

  if (ticks.size() > 1) {
    /* computing beats intervals */
    bpmIntervals.resize(ticks.size() - 1);
    for (uint i = 1; i < ticks.size(); i++) {
      bpmIntervals[i-1] = ticks[i] - ticks[i-1];
    }
    /* computing rubato regions */
    //bpmRubato->input("beats").set(ticks);
    //bpmRubato->output("rubatoStart").set(rubatoStart);
    //bpmRubato->output("rubatoStop").set(rubatoStop);
    //bpmRubato->output("rubatoNumber").set(rubatoNumber);
    //bpmRubato->compute();
  }



  // ====================== printing results ==========================

  cout << "bpm: " << bpm << endl;
  cout << "ticks: " << ticks << endl;
  cout << "estimates: " << estimates << endl;
  cout << "bpmIntervals: " << bpmIntervals << endl;
  //cout << "rubatoStart: " << rubatoStart << endl;
  //cout << "rubatoStop: " << rubatoStop << endl;
  //cout << "rubatoNumber:" << rubatoNumber << endl;

  // ====================== clean up ==================================

  delete audioloader;
  delete frameCutter;
  delete window;
  delete fft;
  delete cart2polar;
  delete onsetHfc;
  delete onsetComplex;
  delete spectrum;
  delete tempoTapBands;
  delete tempoTapTicks;
  delete tempoScaleBands;
  delete tempoTap;
  //delete bpmRubato;

  essentia::shutdown();
}
