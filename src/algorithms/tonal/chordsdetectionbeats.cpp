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

#include "chordsdetectionbeats.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* ChordsDetectionBeats::name = "ChordsDetectionBeats";
const char* ChordsDetectionBeats::description = DOC("This algorithm takes the ChordsDetection algorithm from Essentia, and tries to enhance it by using a Beat Tracker for in between estimation.\n"
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

void ChordsDetectionBeats::configure() {
  Real wsize = parameter("windowSize").toReal();
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = parameter("hopSize").toInt();

  // NB: this assumes that frameSize = hopSize * 2, so that we don't have to
  //     require frameSize as well as parameter.
  _numFramesWindow = int((wsize * _sampleRate) / _hopSize) - 1; // wsize = 1.0 , hopSize = 512 --> 85
}

void ChordsDetectionBeats::compute() {
  const vector<vector<Real> >& hpcp = _pcp.get();
  vector<string>& chords = _chords.get();
  vector<Real>& strength = _strength.get();
  const vector<Real>& ticks = _ticks.get(); 
  
  string key;
  string scale;
  Real firstToSecondRelativeStrength;
  Real str; // strength

  chords.reserve(int(hpcp.size()/_numFramesWindow)); // 1478/85 = 17
  //out << "chords.reserve(int(hpcp.size()/_numFramesWindow)); = " << int(hpcp.size()/_numFramesWindow) << endl;
  strength.reserve(int(hpcp.size()/_numFramesWindow));

  //cout << "int(hpcp.size() = " << (int)hpcp.size() << endl; 

  // for (int j=0; j<ticks.size(); j++) {
  //   cout << "tick " << j << " " << ticks[j] << endl;
  // } 
  if(ticks.size() < 2) { 
  throw EssentiaException("Ticks vector should contain at least 2 elements.");
  } 

  Real diffTicks = 0.0f;
  int numFramesTick = 0;
  int initFrame = 0;

  int frameStart=0;
  int frameEnd=0;
  //cout << "ticks.size() = "<<ticks.size()<< "from 0 to "<< ticks.size()-1 << ", ticks[size-1]"<<ticks[ticks.size()-1]<< endl; 
  //cout << "hpcp.size() = length of chords output array in the previous version of the code = " <<hpcp.size()<< endl;

  for (int i = 0; i < ticks.size()-1; ++i){

    diffTicks = ticks[i+1] - ticks[i];
    numFramesTick = int((diffTicks * _sampleRate) / _hopSize);
    frameStart = int((ticks[i] * _sampleRate) / _hopSize);
    frameEnd = frameStart + numFramesTick-1;

    if (frameEnd > hpcp.size()-1) break;

    vector<Real> hpcpMedian = medianFrames(hpcp, frameStart, frameEnd);
    normalize(hpcpMedian);

    _chordsAlgo->input("pcp").set(hpcpMedian);
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

  } // for

  
}//method

} // namespace standard
} // namespace essentia


#include "poolstorage.h"

namespace essentia {
namespace streaming {

const char* ChordsDetectionBeats::name = standard::ChordsDetectionBeats::name;
const char* ChordsDetectionBeats::description = standard::ChordsDetectionBeats::description;

ChordsDetectionBeats::ChordsDetectionBeats() : AlgorithmComposite() {

  declareInput(_pcp, "pcp", "the pitch class profile from which to detect the chord");
  declareOutput(_chords, 1, "chords", "the resulting chords, from A to G");
  declareOutput(_strength, 1, "strength", "the strength of the chord");

  _chordsAlgo = standard::AlgorithmFactory::create("Key");
  _chordsAlgo->configure("profileType", "tonictriad", "usePolyphony", false);
  _poolStorage = new PoolStorage<vector<Real> >(&_pool, "internal.hpcp");

  // FIXME: this is just a temporary hack...
  //        the correct way to do this is to have the algorithm output the chords
  //        continuously while processing, which requires a FrameCutter for vectors
  // Need to set the buffer type to multiple frames as all the chords
  // are output all at once
  _chords.setBufferType(BufferUsage::forMultipleFrames);
  _strength.setBufferType(BufferUsage::forMultipleFrames);

  attach(_pcp, _poolStorage->input("data"));
}

ChordsDetectionBeats::~ChordsDetectionBeats() {
  delete _chordsAlgo;
  delete _poolStorage;
}

void ChordsDetectionBeats::configure() {
  Real wsize = parameter("windowSize").toReal();
  Real sampleRate = parameter("sampleRate").toReal();
  int hopSize = parameter("hopSize").toInt();

  // NB: this assumes that frameSize = hopSize * 2, so that we don't have to
  //     require frameSize as well as parameter.
  _numFramesWindow = int((wsize * sampleRate) / hopSize) - 1;
}

AlgorithmStatus ChordsDetectionBeats::process() {
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

    //vector<Real> hpcpAverage = medianFrames(hpcp, indexStart, indexEnd);
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

void ChordsDetectionBeats::reset() {
  AlgorithmComposite::reset();
  _chordsAlgo->reset();
}


} // namespace streaming
} // namespace essentia
