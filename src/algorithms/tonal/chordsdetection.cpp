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

#include "chordsdetection.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* ChordsDetection::name = "ChordsDetection";
const char* ChordsDetection::description = DOC("Using pitch profile classes, this algorithm calculates the best matching major or minor triad and outputs the result as a string (e.g. A#, Bm, G#m, C). This algorithm uses the Sharp versions of each Flatted note (i.e. Bb -> A#).\n"
"\n"
"Note:\n"
"  - This algorithm assumes that input pcps have been computed with framesize = 2*hopsize\n"
"\n"
"Quality: experimental (prone to errors, algorithm needs improvement)\n"
"\n"
"References:\n"
"  [1] E. Gómez, \"Tonal Description of Polyphonic Audio for Music Content\n"
"  Processing,\" INFORMS Journal on Computing, vol. 18, no. 3, pp. 294–304,\n"
"  2006.\n\n"
"  [2] D. Temperley, \"What's key for key? The Krumhansl-Schmuckler\n"
"  key-finding algorithm reconsidered\", Music Perception vol. 17, no. 1,\n"
"  pp. 65-100, 1999.");

void ChordsDetection::configure() {
  Real wsize = parameter("windowSize").toReal();
  Real sampleRate = parameter("sampleRate").toReal();
  int hopSize = parameter("hopSize").toInt();

  // NB: this assumes that frameSize = hopSize * 2, so that we don't have to
  //     require frameSize as well as parameter.
  _numFramesWindow = int((wsize * sampleRate) / hopSize) - 1;
}

void ChordsDetection::compute() {
  const vector<vector<Real> >& hpcp = _pcp.get();
  vector<string>& chords= _chords.get();
  vector<Real>& strength= _strength.get();

  string key;
  string scale;
  Real firstToSecondRelativeStrength;
  Real str; // strength

  chords.reserve(int(hpcp.size()/_numFramesWindow));
  strength.reserve(int(hpcp.size()/_numFramesWindow));

  for (int i=0; i<int(hpcp.size()); ++i) {

    int indexStart = max(0, i - _numFramesWindow/2);
    int indexEnd = min(i + _numFramesWindow/2, (int)hpcp.size());

    vector<Real> hpcpAverage = meanFrames(hpcp, indexStart, indexEnd);
    normalize(hpcpAverage);

    _chordsAlgo->input("pcp").set(hpcpAverage);
    _chordsAlgo->output("key").set(key);
    _chordsAlgo->output("scale").set(scale);
    _chordsAlgo->output("strength").set(str);
    _chordsAlgo->output("firstToSecondRelativeStrength").set(firstToSecondRelativeStrength);
    _chordsAlgo->compute();

    if (scale == "minor") {
      chords.push_back(key + 'm');
    }
    else {
      chords.push_back(key);
    }

    strength.push_back(str);

  }
}

} // namespace standard
} // namespace essentia


#include "poolstorage.h"

namespace essentia {
namespace streaming {

const char* ChordsDetection::name = standard::ChordsDetection::name;
const char* ChordsDetection::description = standard::ChordsDetection::description;

ChordsDetection::ChordsDetection() : AlgorithmComposite() {

  declareInput(_pcp, "pcp", "the pitch class profile from which to detect the chord");
  declareOutput(_chords, 1, "chords", "the resulting chords, from A to G");
  declareOutput(_strength, 1, "strength", "the strength of the chord");

  _chordsAlgo = standard::AlgorithmFactory::create("Key");
  _chordsAlgo->configure("profileType", "tonictriad", "usePolyphony", false);
  _poolStorage = new PoolStorage<vector<Real> >(&_pool, "internal.hpcp");

  // FIXME: this is just a temporary hack...
  //        the correct way to do this is to have the algorithm output the chords
  //        continuously while processing, which requires a FrameCutter for vectors
  _chords.setBufferType(BufferUsage::forLargeAudioStream);
  _strength.setBufferType(BufferUsage::forLargeAudioStream);
  // Some old buffer settings that were not enough for long audio 
  //BufferInfo binfo;
  //binfo.size = 16384;
  //binfo.maxContiguousElements = 0;
  //_chords.setBufferInfo(binfo);
  //_strength.setBufferInfo(binfo);

  attach(_pcp, _poolStorage->input("data"));
}

ChordsDetection::~ChordsDetection() {
  delete _chordsAlgo;
  delete _poolStorage;
}

void ChordsDetection::configure() {
  Real wsize = parameter("windowSize").toReal();
  Real sampleRate = parameter("sampleRate").toReal();
  int hopSize = parameter("hopSize").toInt();

  // NB: this assumes that frameSize = hopSize * 2, so that we don't have to
  //     require frameSize as well as parameter.
  _numFramesWindow = int((wsize * sampleRate) / hopSize) - 1;
}

AlgorithmStatus ChordsDetection::process() {
  if (!shouldStop()) return PASS;

  const vector<vector<Real> >& hpcp = _pool.value<vector<vector<Real> > >("internal.hpcp");
  string key;
  string scale;
  Real strength;
  Real firstToSecondRelativeStrength;

  // This is very strange, because we jump by a single frame each time, not by
  // the defined windowSize. Is that the expected behavior or is it a bug?
  // eaylon: windowSize is not intended for advancing, but for searching
  // nwack: maybe it could be a smart idea to jump from 1 beat to another instead
  //        of a fixed amount a time (arbitrary frame size)

  for (int i=0; i<(int)hpcp.size(); i++) {

    int indexStart = max(0, i - _numFramesWindow/2);
    int indexEnd = min(i + _numFramesWindow/2, (int)hpcp.size());

    vector<Real> hpcpAverage = meanFrames(hpcp, indexStart, indexEnd);
    normalize(hpcpAverage);

    _chordsAlgo->input("pcp").set(hpcpAverage);
    _chordsAlgo->output("key").set(key);
    _chordsAlgo->output("scale").set(scale);
    _chordsAlgo->output("strength").set(strength);
    _chordsAlgo->output("firstToSecondRelativeStrength").set(firstToSecondRelativeStrength);
    _chordsAlgo->compute();

    if (scale == "minor") {
      _chords.push(key + 'm');
    }
    else {
      _chords.push(key);
    }

    _strength.push(strength);
  }

  return FINISHED;
}

void ChordsDetection::reset() {
  AlgorithmComposite::reset();
  _chordsAlgo->reset();
}


} // namespace streaming
} // namespace essentia
