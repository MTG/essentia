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

#include "singlebeatloudness.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* SingleBeatLoudness::name = "SingleBeatLoudness";
const char* SingleBeatLoudness::description = DOC("This algorithm computes the loudness of a single beat, on the whole frequency range and on each specified frequency band (bands by default: 0-200 Hz, 200-400 Hz, 400-800 Hz, 800-1600 Hz, 1600-3200 Hz, 3200-22000Hz, following E. Scheirer [1]). See the Loudness algorithm for a description of loudness.\n"
"\n"
"This algorithm throws an exception either when parameter beatDuration is larger than beatWindowSize or when the size of the input beat is less than beatWindowSize plus beatDuration.\n"
"\n"
"References:\n"
"  [1] E. D. Scheirer, \"Tempo and beat analysis of acoustic musical signals,\"\n"
"  The Journal of the Acoustical Society of America, vol. 103, p. 588, 1998.\n");


void SingleBeatLoudness::configure() {
  int sampleRate = parameter("sampleRate").toInt();
  _beatWindowSize = int(parameter("beatWindowDuration").toReal() * (Real)sampleRate);
  _beatDuration = int(parameter("beatDuration").toReal() * (Real)sampleRate);
  _peakEnergy = (parameter("onsetStart").toString()=="peakEnergy");

  if (_beatDuration > _beatWindowSize) {
    throw EssentiaException("Parameter beatDuration cannot be larger than beatWindowDuration");
  }

  if (_beatDuration % 2 == 1) _beatDuration++; // as essentia::FFT only runs on even sizes so far. Needs be removed whenever fft will output the whole spectrum

  if (_beatDuration > _beatWindowSize) {
    throw EssentiaException("SingleBeatLoudness: Parameter beatDuration cannot be larger than beatWindowDuration");
  }

  if (_beatDuration % 2 == 1) _beatDuration++; // as essentia::FFT only runs on even sizes so far. Needs be removed whenever fft will output the whole spectrum

  _window->input("frame").set(_beatWindow);
  _window->output("frame").set(_windowedBeat);
  _spectrum->input("frame").set(_windowedBeat);
  _spectrum->output("spectrum").set(_spec);
  _energy->input("array").set(_spec);
  _energy->output("energy").set(_energyValue);

  _frequencyBands = parameter("frequencyBands").toVectorReal();
  int nBands = _frequencyBands.size();
  _energyBand.resize(nBands-1);
  _energyBandValue.resize(nBands-1);
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  for (int i=0; i<nBands-1; i++) {
    _energyBand[i] = factory.create("EnergyBandRatio",
                                    "sampleRate", sampleRate,
                                    "startFrequency", _frequencyBands[i],
                                    "stopFrequency",  _frequencyBands[i+1]);

    _energyBand[i]->input("spectrum").set(_spec);
    _energyBand[i]->output("energyBandRatio").set(_energyBandValue[i]);
  }
}

void SingleBeatLoudness::compute() {
  const vector<Real>& beat = _beat.get();
  Real& loudness = _loudness.get();
  vector<Real>& loudnessBand = _loudnessBand.get();

  if (int(beat.size()) < _beatWindowSize + _beatDuration) {
    throw EssentiaException("SingleBeatLoudness: size of input beat cannot be smaller than beatWindowSize + beatDuration");
  }

  // first find the max peak of energy in the window size, this will be the
  // beginning of our beat
  int argmax = 0;
  Real maxValue = 0.0;
  if (!_peakEnergy) {
    for (int i=0; i<_beatWindowSize; i++) {
      Real power = beat[i] * beat[i];
      if (power > maxValue) {
        argmax = i;
        maxValue = power;
      }
    }
  }
  else {
    vector<Real> beatPower(beat.size());
    for (int i=0; i<(int)beat.size(); i++)
      beatPower[i] = beat[i]*beat[i];
    for (int i=0; i<_beatWindowSize; i++) {
      int endOfBeat = i+_beatDuration;
      Real power = 0;
      for (int j=i; j<endOfBeat; j++) {
        power += beatPower[j];
      }
      if (power > maxValue) {
        maxValue = power;
        argmax = i;
      }
    }
  }

  // copy only the beat and compute its energies
  _beatWindow.resize(_beatDuration);
  for (int i=0; i<_beatDuration; i++) {
    _beatWindow[i] = beat[argmax + i];
  }

  _window->compute();
  _spectrum->compute();
  _energy->compute();
  for (int i=0; i<(int)_energyBand.size(); i++) _energyBand[i]->compute();

  // output values
  loudness        = _energyValue;
  loudnessBand    = _energyBandValue;
}
