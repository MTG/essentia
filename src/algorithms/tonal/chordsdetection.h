/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#ifndef ESSENTIA_CHORDSDETECTION_H
#define ESSENTIA_CHORDSDETECTION_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class ChordsDetection : public Algorithm {

  protected:
    Input<std::vector<std::vector<Real> > > _pcp;
    Output<std::vector<std::string> > _chords;
    Output<std::vector<Real> > _strength;

    Algorithm* _chordsAlgo;
    int _numFramesWindow;

 public:
  ChordsDetection() {

    _chordsAlgo = AlgorithmFactory::create("Key");
    _chordsAlgo->configure("profileType", "tonictriad", "usePolyphony", false);

    declareInput(_pcp, "pcp", "the pitch class profile from which to detect the chord");
    declareOutput(_chords, "chords", "the resulting chords, from A to G");
    declareOutput(_strength, "strength", "the strength of the chord");
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("windowSize", "the size of the window on which to estimate the chords [s]", "(0,inf)", 2.0);
    declareParameter("hopSize", "the hop size with which the input PCPs were computed", "(0,inf)", 2048);
  }

 ~ChordsDetection() {
   delete _chordsAlgo;
 }

  void configure();

  void compute();

  static const char* name;
  static const char* description;

};


} // namespace standard
} // namespace essentia


#include "streamingalgorithmcomposite.h"
#include "pool.h"

namespace essentia {
namespace streaming {

/**
 * @todo make this algo smarter, and make it output chords as soon as they
 *       can be computed, not only at the end...
 */
class ChordsDetection : public AlgorithmComposite {
 protected:
  SinkProxy<std::vector<Real> > _pcp;

  Source<std::string> _chords;
  Source<Real> _strength;

  Pool _pool;
  Algorithm* _poolStorage;
  standard::Algorithm* _chordsAlgo;
  int _numFramesWindow;

 public:
  ChordsDetection();
  ~ChordsDetection();

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("windowSize", "the size of the window on which to estimate the chords [s]", "(0,inf)", 2.0);
    declareParameter("hopSize", "the hop size with which the input PCPs were computed", "(0,inf)", 2048);
  }

  void declareProcessOrder() {
    declareProcessStep(ChainFrom(_poolStorage));
    declareProcessStep(SingleShot(this));
  }

  void configure();
  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;

};


} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CHORDSDETECTION_H
