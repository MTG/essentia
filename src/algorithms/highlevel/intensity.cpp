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

#include "intensity.h"
#include "pool.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Intensity::name = "Intensity";
const char* Intensity::description = DOC("This algorithm classifies the input audio signal as either relaxed (-1), moderate (0), or aggressive (1).\n"
"\n"
"Quality: outdated (non-reliable, poor accuracy).\n"
"\n"
"An exception is thrown if empty input is provided because the \"intensity\" is not defined for that case.");

enum IntensityClass {
  RELAXED = -1,
  MODERATE = 0,
  AGGRESSIVE = 1
};

void Intensity::configure() {
  int sampleRate = parameter("sampleRate").toInt();

  _spectralComplexity->configure("sampleRate", sampleRate);
  _rollOff->configure("sampleRate", sampleRate);
  _spectralPeaks->configure("sampleRate", sampleRate);
}

void Intensity::compute() {
  // prepare computation
  const vector<Real>& signal = _signal.get();

  vector<Real> frame;
  _frameCutter->input("signal").set(signal);
  _frameCutter->output("frame").set(frame);

  vector<Real> windowedFrame;
  _windowing->input("frame").set(frame);
  _windowing->output("frame").set(windowedFrame);

  vector<Real> spectrum;
  _spectrum->input("frame").set(windowedFrame);
  _spectrum->output("spectrum").set(spectrum);

  Real spectralComplexity;
  _spectralComplexity->input("spectrum").set(spectrum);
  _spectralComplexity->output("spectralComplexity").set(spectralComplexity);

  vector<Real> spectralCentralMoments;
  _centralMoments->input("array").set(spectrum);
  _centralMoments->output("centralMoments").set(spectralCentralMoments);

  Real spectralKurtosis, spread, skewness;
  _distributionShape->input("centralMoments").set(spectralCentralMoments);
  _distributionShape->output("kurtosis").set(spectralKurtosis);
  _distributionShape->output("spread").set(spread);
  _distributionShape->output("skewness").set(skewness);

  Real spectralRollOff;
  _rollOff->input("spectrum").set(spectrum);
  _rollOff->output("rollOff").set(spectralRollOff);

  vector<Real> spectralPeakMags, spectralPeakFreqs;
  _spectralPeaks->input("spectrum").set(spectrum);
  _spectralPeaks->output("magnitudes").set(spectralPeakMags);
  _spectralPeaks->output("frequencies").set(spectralPeakFreqs);

  Real dissonance;
  _dissonance->input("frequencies").set(spectralPeakFreqs);
  _dissonance->input("magnitudes").set(spectralPeakMags);
  _dissonance->output("dissonance").set(dissonance);

  Pool p;

  // get first audio frame
  _frameCutter->compute();

  if (frame.empty()) {
    throw EssentiaException("Intensity: the intensity of empty input is undefined.");
  }

  // compute descriptors
  while (!frame.empty()) {
    _windowing->compute();
    _spectrum->compute();
    _spectralComplexity->compute();
    _centralMoments->compute();
    _distributionShape->compute();
    _rollOff->compute();
    _spectralPeaks->compute();
    _dissonance->compute();

    p.add("spectral.complexity", spectralComplexity);
    p.add("spectral.kurtosis", spectralKurtosis);
    p.add("spectral.rollOff", spectralRollOff);
    p.add("signal.dissonance", dissonance);

    _frameCutter->compute();
  }

  // Configure aggregation
  Pool aggStats;
  Algorithm* poolAgg = AlgorithmFactory::create("PoolAggregator");
  const char* statsToCompute[] = {"mean", "dmean", "dmean2"};
  poolAgg->configure("defaultStats", arrayToVector<string>(statsToCompute));
  poolAgg->input("input").set(p);
  poolAgg->output("output").set(aggStats);

  // aggregate the descriptors
  poolAgg->compute();
  delete poolAgg;

  // classify intensity
  int& intensity = _intensity.get();

  if (aggStats.value<Real>("spectral.complexity.mean") <= 12.717778) {
    if (aggStats.value<Real>("spectral.complexity.dmean") <= 1.912363) {
      intensity = RELAXED;
    }
    else {
      if (aggStats.value<Real>("spectral.kurtosis.mean") <= 7.098977) {
        if (aggStats.value<Real>("spectral.rollOff.mean") <= 2046.564331) {
          intensity = RELAXED;
        }
        else {
          intensity = MODERATE;
        }
      }
      else {
        intensity = RELAXED;
      }
    }
  }
  else {
    if (aggStats.value<Real>("signal.dissonance.dmean2") <= 0.04818) {
      intensity = AGGRESSIVE;
    }
    else {
      intensity = MODERATE;
    }
  }
}
