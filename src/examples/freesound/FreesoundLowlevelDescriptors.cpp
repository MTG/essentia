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

#include "FreesoundLowlevelDescriptors.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;

const string FreesoundLowlevelDescriptors::nameSpace="lowlevel.";  

void FreesoundLowlevelDescriptors::createNetwork(SourceBase& source, Pool& pool){

  Real analysisSampleRate =  44100;// TODO: unify

  AlgorithmFactory& factory = AlgorithmFactory::instance();


  Real sampleRate = 44100;
  int frameSize =   2048;
  int hopSize =     1024;
  int zeroPadding = 0;

  string silentFrames ="noise";
  string windowType = "blackmanharris62";

  // FrameCutter
  Algorithm* fc = factory.create("FrameCutter",
                                 "frameSize", frameSize,
                                 "hopSize", hopSize,
                                 "silentFrames", silentFrames);
  source >> fc->input("signal");


  // Silence Rate
  Real thresholds_dB[] = { -20, -30, -60 };

  vector<Real> thresholds(ARRAY_SIZE(thresholds_dB));
  for (uint i=0; i<thresholds.size(); i++) {
    thresholds[i] = db2lin(thresholds_dB[i]/2.0);
  }

  Algorithm* sr = factory.create("SilenceRate","thresholds", thresholds);
  fc->output("frame") >> sr->input("frame");
  sr->output("threshold_0") >> PC(pool, nameSpace + "silence_rate_20dB");
  sr->output("threshold_1") >> PC(pool, nameSpace + "silence_rate_30dB");
  sr->output("threshold_2") >> PC(pool, nameSpace + "silence_rate_60dB");


  // Windowing
  Algorithm* w = factory.create("Windowing",
                                "type", windowType,
                                "zeroPadding", zeroPadding);
  fc->output("frame") >> w->input("frame");


  // Spectrum
  Algorithm* spec = factory.create("Spectrum");
  w->output("frame") >> spec->input("frame");

  
  // Temporal Descriptors
  Algorithm* zcr = factory.create("ZeroCrossingRate");
  fc->output("frame") >> zcr->input("signal");
  zcr->output("zeroCrossingRate") >> PC(pool, nameSpace + "zerocrossingrate");


  // MFCC
  Algorithm* mfcc = factory.create("MFCC");
  spec->output("spectrum") >> mfcc->input("spectrum");
  mfcc->output("bands") >> NOWHERE;
  mfcc->output("mfcc") >> PC(pool, nameSpace + "mfcc");


  // Spectral Decrease
  Algorithm* square = factory.create("UnaryOperator", "type", "square");
  Algorithm* decrease = factory.create("Decrease",
                                       "range", analysisSampleRate * 0.5);
  spec->output("spectrum") >> square->input("array");
  square->output("array") >> decrease->input("array");
  decrease->output("decrease") >> PC(pool, nameSpace + "spectral_decrease");


  // Spectral Energy
  Algorithm* energy = factory.create("Energy");
  spec->output("spectrum") >> energy->input("array");
  energy->output("energy") >> PC(pool, nameSpace + "spectral_energy");

  // Spectral Energy Band Ratio

  Algorithm* ebr_low = factory.create("EnergyBand",
                                      "startCutoffFrequency", 20.0,
                                      "stopCutoffFrequency", 150.0);
  spec->output("spectrum") >> ebr_low->input("spectrum");
  ebr_low->output("energyBand") >> PC(pool, nameSpace + "spectral_energyband_low");

  Algorithm* ebr_mid_low = factory.create("EnergyBand",
                                          "startCutoffFrequency", 150.0,
                                          "stopCutoffFrequency", 800.0);
  spec->output("spectrum") >> ebr_mid_low->input("spectrum");
  ebr_mid_low->output("energyBand") >> PC(pool, nameSpace + "spectral_energyband_middle_low");

  Algorithm* ebr_mid_hi = factory.create("EnergyBand",
                                         "startCutoffFrequency", 800.0,
                                         "stopCutoffFrequency", 4000.0);
  spec->output("spectrum") >> ebr_mid_hi->input("spectrum");
  ebr_mid_hi->output("energyBand") >> PC(pool, nameSpace + "spectral_energyband_middle_high");


  Algorithm* ebr_hi = factory.create("EnergyBand",
                                     "startCutoffFrequency", 4000.0,
                                     "stopCutoffFrequency", 20000.0);
  spec->output("spectrum") >> ebr_hi->input("spectrum");
  ebr_hi->output("energyBand") >> PC(pool, nameSpace + "spectral_energyband_high");


  // Spectral HFC
  Algorithm* hfc = factory.create("HFC");
  spec->output("spectrum") >> hfc->input("spectrum");
  hfc->output("hfc") >> PC(pool, nameSpace + "hfc");


  // Spectral Frequency Bands
  Algorithm* fb = factory.create("FrequencyBands",
                                 "sampleRate", analysisSampleRate);
  spec->output("spectrum") >> fb->input("spectrum");
  fb->output("bands") >> PC(pool, nameSpace + "frequency_bands");


  // Spectral RMS
  Algorithm* rms = factory.create("RMS");
  spec->output("spectrum") >> rms->input("array");
  rms->output("rms") >> PC(pool, nameSpace + "spectral_rms");


  // Spectral Flux
  Algorithm* flux = factory.create("Flux");
  spec->output("spectrum") >> flux->input("spectrum");
  flux->output("flux") >> PC(pool, nameSpace + "spectral_flux");


  // Spectral Roll Off
  Algorithm* ro = factory.create("RollOff");
  spec->output("spectrum") >> ro->input("spectrum");
  ro->output("rollOff") >> PC(pool, nameSpace + "spectral_rolloff");


  // Spectral Strong Peak
  Algorithm* sp = factory.create("StrongPeak");
  spec->output("spectrum") >> sp->input("spectrum");
  sp->output("strongPeak") >> PC(pool, nameSpace + "spectral_strongpeak");


  // BarkBands
  uint nBarkBands = 27;
  Algorithm* barkBands = factory.create("BarkBands",
                                        "numberBands", nBarkBands);
  spec->output("spectrum") >> barkBands->input("spectrum");
  barkBands->output("bands") >> PC(pool, nameSpace + "barkbands");


  // Spectral Crest
  Algorithm* crest = factory.create("Crest");
  barkBands->output("bands") >> crest->input("array");
  crest->output("crest") >> PC(pool, nameSpace + "spectral_crest");


  // Spectral Flatness DB
  Algorithm* flatness = factory.create("FlatnessDB");
  barkBands->output("bands") >> flatness->input("array");
  flatness->output("flatnessDB") >> PC(pool, nameSpace + "spectral_flatness_db");


  // Spectral Centroid
  Algorithm* square2 = factory.create("UnaryOperator", "type", "square");
  Algorithm* centroid = factory.create("Centroid",
                                       "range", analysisSampleRate * 0.5);
  spec->output("spectrum") >> square2->input("array");
  square2->output("array") >> centroid->input("array");
  centroid->output("centroid") >> PC(pool, nameSpace + "spectral_centroid");


  // Spectral Central Moments Statistics
  Algorithm* cm = factory.create("CentralMoments",
                                 "range", analysisSampleRate * 0.5);
  Algorithm* ds = factory.create("DistributionShape");
  spec->output("spectrum") >> cm->input("array");
  cm->output("centralMoments") >> ds->input("centralMoments");
  ds->output("kurtosis") >> PC(pool, nameSpace + "spectral_kurtosis");
  ds->output("spread") >> PC(pool, nameSpace + "spectral_spread");
  ds->output("skewness")>> PC(pool, nameSpace + "spectral_skewness");


  // Spectral Dissonance
  Algorithm* peaks = factory.create("SpectralPeaks",
                                    "orderBy", "frequency");
  Algorithm* diss = factory.create("Dissonance");
  spec->output("spectrum") >> peaks->input("spectrum");
  peaks->output("frequencies") >> diss->input("frequencies");
  peaks->output("magnitudes") >> diss->input("magnitudes");
  diss->output("dissonance") >> PC(pool, nameSpace + "dissonance");

  // Spectral Contrast
  Algorithm* sc = factory.create("SpectralContrast",
                                 "frameSize", frameSize,
                                 "sampleRate", analysisSampleRate,
                                 "numberBands", 6,
                                 "lowFrequencyBound", 20,
                                 "highFrequencyBound", 11000,
                                 "neighbourRatio", 0.4,
                                 "staticDistribution", 0.15);

  spec->output("spectrum") >> sc->input("spectrum");
  sc->output("spectralContrast") >> PC(pool, nameSpace + "spectral_contrast");
  sc->output("spectralValley") >> PC(pool, nameSpace + "scvalleys");


  // Spectral BarkBands Central Moments Statistics
  Algorithm* bbcm = factory.create("CentralMoments",
                                 "range", nBarkBands-1);
  Algorithm* ds2 = factory.create("DistributionShape");
  barkBands->output("bands") >> bbcm->input("array");
  bbcm->output("centralMoments") >> ds2->input("centralMoments");
  ds2->output("kurtosis") >> PC(pool, nameSpace + "barkbands_kurtosis");
  ds2->output("spread") >> PC(pool, nameSpace + "barkbands_spread");
  ds2->output("skewness") >> PC(pool, nameSpace + "barkbands_skewness");


  // Spectral Complexity
  Algorithm* tc = factory.create("SpectralComplexity",
                                 "magnitudeThreshold", 0.005);
  spec->output("spectrum") >> tc->input("spectrum");
  tc->output("spectralComplexity") >> PC(pool, nameSpace + "spectral_complexity");


  // Pitch Detection
  Algorithm* pitch = factory.create("PitchYinFFT",
                                    "frameSize", frameSize);
  spec->output("spectrum") >> pitch->input("spectrum");
  pitch->output("pitch") >> PC(pool, nameSpace + "pitch");
  pitch->output("pitchConfidence") >> PC(pool, nameSpace + "pitch_instantaneous_confidence");


  // Pitch Salience
  Algorithm* ps = factory.create("PitchSalience");
  spec->output("spectrum") >> ps->input("spectrum");
  ps->output("pitchSalience") >> PC(pool, nameSpace + "pitch_salience");


  // Loudness
  Algorithm* ln = factory.create("Loudness");
  fc->output("frame") >> ln->input("signal");
  ln->output("loudness") >> PC(pool, nameSpace + "loudness");

  // StartStopSilence
  Algorithm* ss = factory.create("StartStopSilence","threshold",-60);
  fc->output("frame") >> ss->input("frame");
  ss->output("startFrame") >> PC(pool, nameSpace + "startFrame");
  ss->output("stopFrame") >> PC(pool, nameSpace + "stopFrame");
} 

Real squeezeRange(Real& x, Real& x1, Real& x2);

Real squeezeRange(Real& x, Real& x1, Real& x2) {
  return (0.5 + 0.5 * tanh(-1.0 + 2.0 * (x - x1) / (x2 - x1)));
}


void FreesoundLowlevelDescriptors::computeAverageLoudness(Pool& pool){ // after computing network

  vector<Real> levelArray = pool.value<vector<Real> >(nameSpace + "loudness");
  pool.remove(nameSpace + "loudness");

  // Maximum dynamic
  Real EPSILON = 10e-5;
  Real maxValue = levelArray[argmax(levelArray)];
  if (maxValue <= EPSILON) {
    maxValue = EPSILON;
  }

  // Normalization to the maximum
  Real THRESHOLD = 0.0001; // this corresponds to -80dB
  for (uint i=0; i<levelArray.size(); i++) {
    levelArray[i] /= maxValue;
    if (levelArray[i] <= THRESHOLD) {
      levelArray[i] = THRESHOLD;
    }
  }

  // Average Level
  Real levelAverage = pow2db(mean(levelArray));

  // Re-scaling and range-control
  // This yields in numbers between
  // 0 for signals with  large dynamic variance and thus low dynamic average
  // 1 for signal with little dynamic range and thus
  // a dynamic average close to the maximum
  Real x1 = -5.0;
  Real x2 = -2.0;
  Real levelAverageSqueezed = squeezeRange(levelAverage, x1, x2);
  pool.set(nameSpace + "average_loudness", levelAverageSqueezed);
}



