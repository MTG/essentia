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
    Real _sampleRate; 
    int _hopSize;
    std::string _chromaPick;

  public:
    ChordsDetectionBeats() {

      _chordsAlgo = AlgorithmFactory::create("Key");
      _chordsAlgo->configure("profileType", "tonictriad", "usePolyphony", false);

      declareInput(_pcp, "pcp", "the pitch class profile from which to detect the chord");
      declareInput(_ticks, "ticks", "the list of beat positions (in seconds)");
      declareOutput(_chords, "chords", "the resulting chords, from A to G");
      declareOutput(_strength, "strength", "the strength of the chords");
    }

    void declareParameters() {
      declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
      declareParameter("hopSize", "the hop size with which the input PCPs were computed", "(0,inf)", 2048);
      declareParameter("chromaPick", "method of calculating singleton chroma for interbeat interval", "{starting_beat,interbeat_median}", "interbeat_median");
    }

    ~ChordsDetectionBeats() {
     delete _chordsAlgo;
   }

    void configure();

    void compute();

    static const char* name;
    static const char* category;    
    static const char* description;
};


} // namespace standard
} // namespace essentia


#endif // ESSENTIA_CHORDSDETECTIONBEATS_H
