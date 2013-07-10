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

#ifndef ESSENTIA_CHORDSDESCRIPTORS_H
#define ESSENTIA_CHORDSDESCRIPTORS_H

#include "algorithmfactory.h"

namespace essentia {
namespace standard {

class ChordsDescriptors : public Algorithm {

 protected:
  Input<std::vector<std::string> > _chords;
  Input<std::string> _key;
  Input<std::string> _scale;

  Output<std::vector<Real> > _chordsHistogram;
  Output<Real> _chordsNumberRate;
  Output<Real> _chordsChangesRate;
  Output<std::string> _chordsKey;
  Output<std::string> _chordsScale;

  static const char* circleOfFifth[];
  int chordIndex(const std::string& chord);
  std::map<std::string, Real> chordsHistogram(const std::vector<std::string>& chords);
  std::map<std::string, Real> chordsHistogramNorm(std::map<std::string, Real>& histogram,
                                                  const std::string& key);

 public:

  ChordsDescriptors() {
    declareInput(_chords, "chords", "the chord progression");
    declareInput(_key, "key", "the key of the whole song, from A to G");
    declareInput(_scale, "scale", "the scale of the whole song (major or minor)");

    declareOutput(_chordsHistogram, "chordsHistogram", "the normalized histogram of chords");
    declareOutput(_chordsNumberRate, "chordsNumberRate", "the ratio of different chords from the total number of chords in the progression");
    declareOutput(_chordsChangesRate, "chordsChangesRate", "the rate at which chords change in the progression");
    declareOutput(_chordsKey, "chordsKey", "the most frequent chord of the progression");
    declareOutput(_chordsScale, "chordsScale", "the scale of the most frequent chord of the progression (either 'major' or 'minor')");
  }

  void declareParameters() {}
  void configure() {}

  void compute();

  static const char* name;
  static const char* description;

};

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

class ChordsDescriptors : public Algorithm {

 protected:
  Sink<std::string> _chords;
  Sink<std::string> _key;
  Sink<std::string> _scale;

  Source<std::vector<Real> > _chordsHistogram;
  Source<Real> _chordsNumberRate;
  Source<Real> _chordsChangesRate;
  Source<std::string> _chordsKey;
  Source<std::string> _chordsScale;

  std::vector<std::string> _accu;

  standard::Algorithm* _chordsAlgo;

 public:

  ChordsDescriptors() {
    declareInput(_chords, 1, "chords", "the chord progression");
    declareInput(_key, 1, "key", "the key of the whole song, from A to G");
    declareInput(_scale, 1, "scale", "the scale of the whole song (major or minor)");

    declareOutput(_chordsHistogram, 0, "chordsHistogram", "the normalized histogram of chords");
    declareOutput(_chordsNumberRate, 0, "chordsNumberRate", "the ratio of different chords from the total number of chords in the progression");
    declareOutput(_chordsChangesRate, 0, "chordsChangesRate", "the rate at which chords change in the progression");
    declareOutput(_chordsKey, 0, "chordsKey", "the key of the progression, taken as the most frequent chord");
    declareOutput(_chordsScale, 0, "chordsScale", "the scale of the progression, whether major or minor");

    _chordsAlgo = standard::AlgorithmFactory::create("ChordsDescriptors");
  }

  ~ChordsDescriptors() {
    delete _chordsAlgo;
  }

  void declareParameters() {}

  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* description;
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_CHORDSDESCRIPTORS_H
