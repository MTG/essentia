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

#include "FreesoundLowlevelDescriptors.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;

const string FreesoundLowlevelDescriptors::nameSpace="lowlevel.";  

void FreesoundLowlevelDescriptors::createNetwork(SourceBase& source, Pool& pool){

  AlgorithmFactory& factory = AlgorithmFactory::instance();

  Real sampleRate = options.value<Real>("analysisSampleRate");
  int frameSize =   int(options.value<Real>("lowlevel.frameSize"));
  int hopSize =     int(options.value<Real>("lowlevel.hopSize"));
  int zeroPadding = int(options.value<Real>("lowlevel.zeroPadding"));
  string silentFrames = options.value<string>("lowlevel.silentFrames");
  string windowType = options.value<string>("lowlevel.windowType");

  Algorithm* fc = factory.create("FrameCutter",
                                 "frameSize", frameSize,
                                 "hopSize", hopSize,
                                 "silentFrames", silentFrames);
  
  Algorithm* w = factory.create("Windowing",
                                "type", windowType,
                                "zeroPadding", zeroPadding);
  
  Algorithm* spec = factory.create("Spectrum");


  source >> fc->input("signal");
  fc->output("frame") >> w->input("frame");
  w->output("frame") >> spec->input("frame");

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

  // Zero crossing rate
  Algorithm* zcr = factory.create("ZeroCrossingRate");
  fc->output("frame") >> zcr->input("signal");
  zcr->output("zeroCrossingRate") >> PC(pool, nameSpace + "zerocrossingrate");

  // MFCC
  Algorithm* mfcc = factory.create("MFCC");
  spec->output("spectrum") >> mfcc->input("spectrum");
  mfcc->output("bands") >> PC(pool, nameSpace + "melbands");
  mfcc->output("mfcc") >> PC(pool, nameSpace + "mfcc");

  // Spectral MelBands Central Moments Statistics, Flatness and Crest
  Algorithm* mels_cm = factory.create("CentralMoments", "range", 40-1);
  Algorithm* mels_ds = factory.create("DistributionShape");
  mfcc->output("bands")             >> mels_cm->input("array");
  mels_cm->output("centralMoments") >> mels_ds->input("centralMoments");
  mels_ds->output("kurtosis")       >> PC(pool, nameSpace + "melbands_kurtosis");
  mels_ds->output("spread")         >> PC(pool, nameSpace + "melbands_spread");
  mels_ds->output("skewness")       >> PC(pool, nameSpace + "melbands_skewness");

  Algorithm* mels_fl = factory.create("FlatnessDB");
  Algorithm* mels_cr = factory.create("Crest");
  mfcc->output("bands")      >> mels_fl->input("array");
  mfcc->output("bands")      >> mels_cr->input("array");
  mels_fl->output("flatnessDB")  >> PC(pool, nameSpace + "melbands_flatness_db");
  mels_cr->output("crest")       >> PC(pool, nameSpace + "melbands_crest");

  // taken from MusicExtractor. MelBands 128 
  Algorithm* melbands96 = factory.create("MelBands", "numberBands", 96);
  spec->output("spectrum")     >> melbands96->input("spectrum");
  melbands96->output("bands") >> PC(pool, nameSpace + "melbands96");

  // ERB Bands and GFCC
  Algorithm* gfcc = factory.create("GFCC",
                                   "highFrequencyBound", 9795, 
                                   "lowFrequencyBound", 26,
                                   "numberBands", 18);
  spec->output("spectrum") >> gfcc->input("spectrum");
  gfcc->output("bands") >>PC(pool, nameSpace + "erbbands");
  gfcc->output("gfcc") >> PC(pool, nameSpace + "gfcc");

  // Spectral ERBBands Central Moments Statistics, Flatness and Crest
  Algorithm* erbs_cm = factory.create("CentralMoments", "range", 18-1);
  Algorithm* erbs_ds = factory.create("DistributionShape");
  gfcc->output("bands")             >> erbs_cm->input("array");
  erbs_cm->output("centralMoments") >> erbs_ds->input("centralMoments");
  erbs_ds->output("kurtosis")       >> PC(pool, nameSpace + "erbbands_kurtosis");
  erbs_ds->output("spread")         >> PC(pool, nameSpace + "erbbands_spread");
  erbs_ds->output("skewness")       >> PC(pool, nameSpace + "erbbands_skewness");

  Algorithm* erbs_fl = factory.create("FlatnessDB");
  Algorithm* erbs_cr = factory.create("Crest");
  gfcc->output("bands")      >> erbs_fl->input("array");
  gfcc->output("bands")      >> erbs_cr->input("array");
  erbs_fl->output("flatnessDB")  >> PC(pool, nameSpace + "erbbands_flatness_db");
  erbs_cr->output("crest")       >> PC(pool, nameSpace + "erbbands_crest");

 // BarkBands
  uint nBarkBands = 27;
  Algorithm* barkBands = factory.create("BarkBands",
                                        "numberBands", nBarkBands);
  spec->output("spectrum") >> barkBands->input("spectrum");
  barkBands->output("bands") >> PC(pool, nameSpace + "barkbands");

  // Spectral BarkBands Central Moments Statistics, Flatness and Crest
  Algorithm* barks_cm = factory.create("CentralMoments", "range", nBarkBands-1);
  Algorithm* barks_ds = factory.create("DistributionShape");
  barkBands->output("bands")          >> barks_cm->input("array");
  barks_cm->output("centralMoments")  >> barks_ds->input("centralMoments");
  barks_ds->output("kurtosis")        >> PC(pool, nameSpace + "barkbands_kurtosis");
  barks_ds->output("spread")          >> PC(pool, nameSpace + "barkbands_spread");
  barks_ds->output("skewness")        >> PC(pool, nameSpace + "barkbands_skewness");

  Algorithm* barks_fl = factory.create("FlatnessDB");
  Algorithm* barks_cr = factory.create("Crest");
  barkBands->output("bands")      >> barks_fl->input("array");
  barkBands->output("bands")      >> barks_cr->input("array");
  barks_fl->output("flatnessDB")  >> PC(pool, nameSpace + "barkbands_flatness_db");
  barks_cr->output("crest")       >> PC(pool, nameSpace + "barkbands_crest");

  // Spectral Decrease
  Algorithm* square = factory.create("UnaryOperator", "type", "square");
  Algorithm* decrease = factory.create("Decrease",
                                       "range", sampleRate * 0.5);
  spec->output("spectrum") >> square->input("array");
  square->output("array") >> decrease->input("array");
  decrease->output("decrease") >> PC(pool, nameSpace + "spectral_decrease");

  // Spectral Roll Off
  Algorithm* ro = factory.create("RollOff");
  spec->output("spectrum") >> ro->input("spectrum");
  ro->output("rollOff") >> PC(pool, nameSpace + "spectral_rolloff");

  // Spectral Energy
  Algorithm* energy = factory.create("Energy");
  spec->output("spectrum") >> energy->input("array");
  energy->output("energy") >> PC(pool, nameSpace + "spectral_energy");

  // Spectral RMS
  Algorithm* rms = factory.create("RMS");
  spec->output("spectrum") >> rms->input("array");
  rms->output("rms") >> PC(pool, nameSpace + "spectral_rms");

  // Spectral Energy Band Ratio
  Algorithm* ebr_low      = factory.create("EnergyBand",
                                           "startCutoffFrequency", 20.0,
                                           "stopCutoffFrequency", 150.0);
  Algorithm* ebr_mid_low  = factory.create("EnergyBand",
                                           "startCutoffFrequency", 150.0,
                                           "stopCutoffFrequency", 800.0);
  Algorithm* ebr_mid_hi   = factory.create("EnergyBand",
                                           "startCutoffFrequency", 800.0,
                                           "stopCutoffFrequency", 4000.0);
  Algorithm* ebr_hi       = factory.create("EnergyBand",
                                           "startCutoffFrequency", 4000.0,
                                           "stopCutoffFrequency", 20000.0);
  spec->output("spectrum")  >> ebr_low->input("spectrum");
  spec->output("spectrum")  >> ebr_mid_low->input("spectrum");
  spec->output("spectrum")  >> ebr_mid_hi->input("spectrum");
  spec->output("spectrum")  >> ebr_hi->input("spectrum");
  ebr_low->output("energyBand")     >> PC(pool, nameSpace + "spectral_energyband_low");
  ebr_mid_low->output("energyBand") >> PC(pool, nameSpace + "spectral_energyband_middle_low");
  ebr_mid_hi->output("energyBand")  >> PC(pool, nameSpace + "spectral_energyband_middle_high");
  ebr_hi->output("energyBand")      >> PC(pool, nameSpace + "spectral_energyband_high");

  // Spectral HFC
  Algorithm* hfc = factory.create("HFC");
  spec->output("spectrum") >> hfc->input("spectrum");
  hfc->output("hfc") >> PC(pool, nameSpace + "hfc");

  // Spectral Flux
  Algorithm* flux = factory.create("Flux");
  spec->output("spectrum") >> flux->input("spectrum");
  flux->output("flux") >> PC(pool, nameSpace + "spectral_flux");

  // Spectral Strong Peak
  Algorithm* sp = factory.create("StrongPeak");
  spec->output("spectrum") >> sp->input("spectrum");
  sp->output("strongPeak") >> PC(pool, nameSpace + "spectral_strongpeak");

  // Spectral Complexity
  Algorithm* tc = factory.create("SpectralComplexity",
                                 "magnitudeThreshold", 0.005);
  spec->output("spectrum") >> tc->input("spectrum");
  tc->output("spectralComplexity") >> PC(pool, nameSpace + "spectral_complexity");
  
  // Pitch Salience
  Algorithm* ps = factory.create("PitchSalience");
  spec->output("spectrum") >> ps->input("spectrum");
  ps->output("pitchSalience") >> PC(pool, nameSpace + "pitch_salience");

  /*
  // Spectral Frequency Bands
  Algorithm* fb = factory.create("FrequencyBands",
                                 "sampleRate", sampleRate);
  spec->output("spectrum") >> fb->input("spectrum");
  fb->output("bands") >> PC(pool, nameSpace + "frequency_bands");
  */

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
                                       "range", sampleRate * 0.5);
  spec->output("spectrum") >> square2->input("array");
  square2->output("array") >> centroid->input("array");
  centroid->output("centroid") >> PC(pool, nameSpace + "spectral_centroid");

  // Spectral Central Moments Statistics
  Algorithm* cm = factory.create("CentralMoments",
                                 "range", sampleRate * 0.5);
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

  // Spectral Entropy
  Algorithm* ent = factory.create("Entropy");
  spec->output("spectrum") >> ent->input("array");
  ent->output("entropy") >> PC(pool, nameSpace + "spectral_entropy");

  // Spectral Contrast
  Algorithm* sc = factory.create("SpectralContrast",
                                 "frameSize", frameSize,
                                 "sampleRate", sampleRate,
                                 "numberBands", 6,
                                 "lowFrequencyBound", 20,
                                 "highFrequencyBound", 11000,
                                 "neighbourRatio", 0.4,
                                 "staticDistribution", 0.15);

  spec->output("spectrum") >> sc->input("spectrum");
  sc->output("spectralContrast") >> PC(pool, nameSpace + "spectral_contrast_coeffs");
  sc->output("spectralValley") >> PC(pool, nameSpace + "spectral_contrast_valleys");

  // Pitch Detection
  Algorithm* pitch = factory.create("PitchYinFFT",
                                    "frameSize", frameSize);
  spec->output("spectrum") >> pitch->input("spectrum");
  pitch->output("pitch") >> PC(pool, nameSpace + "pitch");
  pitch->output("pitchConfidence") >> PC(pool, nameSpace + "pitch_instantaneous_confidence");

  // Note: loudness is computed on shorter frames compared to MusicExtractor
  Algorithm* ln = factory.create("Loudness");
  fc->output("frame") >> ln->input("signal");
  ln->output("loudness") >> PC(pool, nameSpace + "loudness");

  // StartStopSilence
  Algorithm* ss = factory.create("StartStopSilence","threshold",-60);
  fc->output("frame") >> ss->input("frame");
  ss->output("startFrame") >> PC(pool, nameSpace + "sound_start_frame");
  ss->output("stopFrame") >> PC(pool, nameSpace + "sound_stop_frame");

  // Dynamic complexity
  Algorithm* dc = factory.create("DynamicComplexity", "sampleRate", sampleRate);
  source                          >> dc->input("signal");
  dc->output("dynamicComplexity") >> PC(pool, nameSpace + "dynamic_complexity");
  dc->output("loudness")          >> NOWHERE; // TODO ??? --> should correspond to average_loudness value, if so --> simplify
} 

inline Real squeezeRange(Real& x, Real& x1, Real& x2) {
  return (0.5 + 0.5 * tanh(-1.0 + 2.0 * (x - x1) / (x2 - x1)));
}

void FreesoundLowlevelDescriptors::computeAverageLoudness(Pool& pool) { // after computing network

  // TODO: would this fail on empty signal?
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
