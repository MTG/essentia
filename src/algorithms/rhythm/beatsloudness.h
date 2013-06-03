/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_BEATSLOUDNESS_H
#define ESSENTIA_BEATSLOUDNESS_H

#include "streamingalgorithmcomposite.h"
#include "network.h"

namespace essentia {
namespace streaming {

class BeatsLoudness : public AlgorithmComposite {
 protected:
  SinkProxy<Real> _signal;
  SourceProxy<Real> _loudness;
  SourceProxy<std::vector<Real> > _loudnessBandRatio;

  Algorithm* _slicer, *_beatLoud;

 public:
  BeatsLoudness();

  ~BeatsLoudness() {
    delete _slicer;
    delete _beatLoud;
  }

  void declareParameters() {
    Real defaultBands[] = { 0.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 22000.0 };
    declareParameter("sampleRate", "the audio sampling rate [Hz]",
                     "(0,inf)", 44100.);
    declareParameter("beats", "the list of beat positions (each position is in "
                     "seconds)",
                     "", std::vector<Real>());
    declareParameter("beatWindowDuration", "the duration of the window in "
                     "which to look for the beginning of the beat (centered "
                     "around the positions in 'beats') [s]",
                     "(0,inf)", 0.1);
    // 50ms default value estimation after checking some drums' kicks duration
    // on Freesound
    declareParameter("beatDuration", "the duration of the window in which the "
                     "beat will be restricted [s]",
                     "(0,inf)", 0.05);
    declareParameter("frequencyBands", "the list of bands to compute energy ratios [Hz", "", arrayToVector<Real>(defaultBands));
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_slicer));
  }

  void configure();

  static const char* name;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#include "vectorinput.h"
#include "network.h"
#include "algorithm.h"
#include "pool.h"

namespace essentia {
namespace standard {

class BeatsLoudness : public Algorithm {

 protected:
  Input<std::vector<Real> > _signal;
  Output<std::vector<Real> > _loudness;
  Output<std::vector<std::vector<Real> > > _loudnessBand;

  streaming::Algorithm* _beatLoud;
  streaming::VectorInput<Real>* _vectorInput;
  scheduler::Network* _network;
  Pool _pool;

  void createInnerNetwork();

 public:
  BeatsLoudness() {
    declareInput(_signal, "signal", "the input audio signal");
    declareOutput(_loudness, "loudness", "the beat's energy in the whole spectrum");
    declareOutput(_loudnessBand, "loudnessBandRatio", "the ratio of the beat's energy on each frequency band");
    createInnerNetwork();
  }

  void declareParameters() {
    Real defaultBands[] = { 20.0, 150.0, 400.0, 3200.0, 7000.0, 22000.0};
    declareParameter("sampleRate", "the audio sampling rate [Hz]",
                     "(0,inf)", 44100.);
    declareParameter("beats", "the list of beat positions (each position is in "
                     "seconds)",
                     "", std::vector<Real>());
    declareParameter("beatWindowDuration", "the duration of the window in "
                     "which to look for the beginning of the beat (centered "
                     "around the positions in 'beats') [s]",
                     "(0,inf)", 0.1);
    // 50ms default value estimation after checking some drums' kicks duration
    // on Freesound
    declareParameter("beatDuration", "the duration of the window in which the "
                     "beat will be restricted [s]",
                     "(0,inf)", 0.05);
    declareParameter("frequencyBands", "the list of bands to compute energy ratios [Hz", "", arrayToVector<Real>(defaultBands));
  }

 ~BeatsLoudness();

  void configure();
  void compute();
  void reset() { _network->reset(); }

  static const char* name;
  static const char* description;

};
} // namespace standard
} // namespace essentia


#endif // ESSENTIA_BEATSLOUDNESS_H
