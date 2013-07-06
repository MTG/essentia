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

#ifndef ESSENTIA_SINGLEBEATLOUDNESS_H
#define ESSENTIA_SINGLEBEATLOUDNESS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class SingleBeatLoudness : public Algorithm {

 protected:
  Input<std::vector<AudioSample> > _beat;
  Output<Real> _loudness;
  Output<std::vector<Real> > _loudnessBand;

  int _beatWindowSize;
  int _beatDuration;
  bool _peakEnergy;
  std::vector<Real> _frequencyBands;

  Algorithm* _window, *_spectrum, *_energy;
  std::vector<Real> _beatWindow, _windowedBeat, _spec;
  std::vector<Algorithm*> _energyBand;
  Real _energyValue;
  std::vector<Real> _energyBandValue;

 public:
  SingleBeatLoudness() {
    declareInput(_beat, "beat", "the sliced beat");
    declareOutput(_loudness, "loudness", "the beat's energy in the whole spectrum");
    declareOutput(_loudnessBand, "loudnessBandRatio", "the beat's energy ratio on each band");

   AlgorithmFactory& factory = AlgorithmFactory::instance();
   _window     = factory.create("Windowing", "zeroPhase", false,
                                "type", "blackmanharris62");
   _spectrum   = factory.create("Spectrum");
   _energy     = factory.create("Energy");
  }

  ~SingleBeatLoudness() {
    if (_window) delete _window;
    if (_spectrum) delete _spectrum;
    if (_energy) delete _energy;
    for (int i=0; i<(int)_energyBand.size(); i++) {
      if (_energyBand[i]) delete _energyBand[i];
    }
  }

  void declareParameters() {
    Real defaultBands[] = { 0.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 22000.0 };
    declareParameter("sampleRate", "the audio sampling rate [Hz]", "(0,inf)", 44100.);
    declareParameter("beatWindowDuration", "the size of the window in which to look for the beginning of the beat [s]", "(0,inf)", 0.1);
    declareParameter("beatDuration", "the size of the window in which the beat will be restricted [s]", "(0,inf)", 0.05);
    declareParameter("frequencyBands", "the bands", "", arrayToVector<Real>(defaultBands));
    declareParameter("onsetStart", "criteria for finding the start of the beat", "{sumEnergy, peakEnergy}", "sumEnergy");
  }

  void compute();
  void configure();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class SingleBeatLoudness : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _beat;
  Source<Real> _loudness;
  Source<std::vector<Real> > _loudnessBand;


 public:
  SingleBeatLoudness() {
    declareAlgorithm("SingleBeatLoudness");
    declareInput(_beat,             TOKEN, "beat");
    declareOutput(_loudness,        TOKEN, "loudness");
    declareOutput(_loudnessBand,    TOKEN, "loudnessBandRatio");
  }
};


} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_SINGLEBEATLOUDNESS_H
