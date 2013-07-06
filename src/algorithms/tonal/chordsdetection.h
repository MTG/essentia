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
