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

#ifndef ESSENTIA_CHORDSDETECTIONBEATS_H
#define ESSENTIA_CHORDSDETECTIONBEATS_H

#include "algorithmfactory.h"
#include <list>
#include <iostream>

namespace essentia {
namespace standard {

class ChordsDetectionBeats : public Algorithm {

  protected:
    Input<std::vector<std::vector<Real> > > _pcp;
    Input<std::vector<Real> > _ticks;
    Output<std::vector<std::string> > _chords;
    Output<std::vector<Real> > _strength;

    Algorithm* _chordsAlgo;
    int _numFramesWindow;
    Real _sampleRate; 
    int _hopSize;

  public:
    ChordsDetectionBeats() {

      _chordsAlgo = AlgorithmFactory::create("Key");
      _chordsAlgo->configure("profileType", "tonictriad", "usePolyphony", false);

      declareInput(_pcp, "pcp", "the pitch class profile from which to detect the chord");
      declareInput(_ticks, "ticks", "the ticks where is located the beat of the song");
      declareOutput(_chords, "chords", "the resulting chords, from A to G");
      declareOutput(_strength, "strength", "the strength of the chord");
    }

    void declareParameters() {
      declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
      declareParameter("windowSize", "the size of the window on which to estimate the chords [s]", "(0,inf)", 2.0);
      declareParameter("hopSize", "the hop size with which the input PCPs were computed", "(0,inf)", 2048);
    }

    ~ChordsDetectionBeats() {
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

// TODO: the implementation of the streaming mode is from the old algorithm 
// and it was not changed. Implement the streaming mode for the new chords
// detection algorithm.

class ChordsDetectionBeats : public AlgorithmComposite {
 protected:
  SinkProxy<std::vector<Real> > _pcp;
  //SinkProxy<std::vector<Real> > _ticks; //correct? useless for the moment 

  Source<std::string> _chords;
  Source<Real> _strength;

  Pool _pool;
  Algorithm* _poolStorage;
  standard::Algorithm* _chordsAlgo;
  int _numFramesWindow;
  

 public:
  ChordsDetectionBeats();
  ~ChordsDetectionBeats();

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

#endif // ESSENTIA_CHORDSDETECTIONBEATS_H
