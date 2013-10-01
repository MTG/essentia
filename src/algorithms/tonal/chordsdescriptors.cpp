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

#include "chordsdescriptors.h"
#include "stringutil.h"

using namespace std;

namespace essentia {
namespace standard {

const char* ChordsDescriptors::name = "ChordsDescriptors";
const char* ChordsDescriptors::description = DOC("Given a chord progression this algorithm describes it by means of key, scale, histogram, and rate of change.\n"
"Note:\n"
"  - chordsHistogram indexes follow the circle of fifths order, while being shifted to the input key and scale\n"
"  - key and scale are taken from the most frequent chord. In the case where multiple chords are equally frequent, the chord is hierarchically chosen from the circle of fifths.\n"
"  - valid chords are C, Em, G, Bm, D, F#m, A, C#m, E, G#m, B, D#m, F#, A#m, C#, Fm, G#, Cm, D#, Gm, A#, Dm, F, Am. Chords that not follow this terminology (i.e. Gb) will raise an exception.\n"
"\n"
"Input chords vector may not be empty, otherwise an exception is thrown.\n"
"\n"
"References:\n"
"  [1] Chord progression - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Chord_progression\n\n"
"  [2] Circle of fifths - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Circle_of_fifths");

const char* ChordsDescriptors::circleOfFifth[] = { "C", "Em", "G", "Bm", "D", "F#m", "A", "C#m", "E", "G#m", "B", "D#m", "F#", "A#m", "C#", "Fm", "G#", "Cm", "D#", "Gm", "A#", "Dm", "F", "Am"};



int ChordsDescriptors::chordIndex(const string& chord) {
  for (int i=0; i<int(ARRAY_SIZE(circleOfFifth)); ++i) {
    if (chord == circleOfFifth[i]) {
      return i;
    }
  }
  throw EssentiaException("ChordsDescriptors: Invalid chord: ", chord);
}


map<string, Real> ChordsDescriptors::chordsHistogram(const vector<string>& chords) {
  map<string, Real> histogram;

  // Initialize
  for (int i=0; i<int(ARRAY_SIZE(circleOfFifth)); ++i) {
    histogram[circleOfFifth[i]] = 0.0;
  }

  // Increment
  for (int i=0; i<int(chords.size()); ++i) {
    histogram[chords[i]] += 1.0;
  }

  // Normalize
  for (int i=0; i<int(histogram.size()); ++i) {
    histogram[circleOfFifth[i]] *= 100.0 / (Real)chords.size();
  }

  return histogram;
}


// offset the list of indices making key as the root index (or 0-index)
map<string, Real> ChordsDescriptors::chordsHistogramNorm(map<string, Real>& histogram, const string& key) {
  int keyIndex = chordIndex(key);
  map<string, Real> histogramNorm = histogram;

  for (int i=0; i<int(histogramNorm.size()); ++i) {
    int chordIndex = i - keyIndex;
    if (chordIndex < 0) {
      chordIndex += ARRAY_SIZE(circleOfFifth);
    }
    histogramNorm[circleOfFifth[chordIndex]] = histogram[circleOfFifth[i]];
  }

  return histogramNorm;
}


void ChordsDescriptors::compute() {
  const vector<string>& chords = _chords.get();

  if (chords.empty()) {
    throw EssentiaException("ChordsDescriptors: Chords input empty");
  }

  string key = toUpper(_key.get());
  string scale = toLower(_scale.get());

  if (_scale.get() == "minor") {
    key += "m";
  }

    // Chords Histogram
  map<string, Real> chordsHist = chordsHistogram(chords);
  map<string, Real> chordsHistNorm = chordsHistogramNorm(chordsHist, key);

  vector<Real>& chordsHistNormVect = _chordsHistogram.get();
  chordsHistNormVect.resize(0); // erase anything that was in there
  for (int i=0; i<int(ARRAY_SIZE(circleOfFifth)); ++i) {
    chordsHistNormVect.push_back(chordsHistNorm[circleOfFifth[i]]);
  }

  // Chords Number Rate
  Real& chordNumberRate = _chordsNumberRate.get();
  chordNumberRate = 0.0;
  for (int i=0; i<int(chordsHistNormVect.size()); ++i) {
    if (chordsHistNormVect[i] > 1.0) {
      chordNumberRate += 1.0;
    }
  }
  chordNumberRate /= (Real)chords.size();

  // Chords Changes Rate
  Real& chordChangesRate = _chordsChangesRate.get();
  chordChangesRate = 0.0;
  for (int i=1; i<int(chords.size()); ++i) {
    if (chords[i] != chords[i-1]) {
      chordChangesRate += 1.0;
    }
  }
  chordChangesRate /= (Real)chords.size();

  // Chords Key and Scale = most frequent chord
  string& chordsKey = _chordsKey.get();
  string& chordsScale = _chordsScale.get();
  chordsKey = "A";
  Real maxValue = 0.0;

  for (int i=0; i<int(ARRAY_SIZE(circleOfFifth)); ++i) {
    if (chordsHist[circleOfFifth[i]] > maxValue) {
      maxValue = chordsHist[circleOfFifth[i]];
      chordsKey = circleOfFifth[i];
    }
  }

  bool major = true;
  string::size_type position = chordsKey.find("m");

  if ((position == 1) || (position == 2)) {
    major = false;
  }

  if (major) {
    chordsKey = chordsKey;
    chordsScale = "major";
  }
  else {
    chordsKey = chordsKey.substr(0, position);
    chordsScale = "minor";
  }
}

} // namespace standard
} // namespace essentia



namespace essentia {
namespace streaming {

const char* ChordsDescriptors::name = standard::ChordsDescriptors::name;
const char* ChordsDescriptors::description = standard::ChordsDescriptors::description;


AlgorithmStatus ChordsDescriptors::process() {
  // this could be implemented as a composite, but it is kept like this for
  // historical and demonstrations reasons
  while (_chords.acquire(1)) {
    _accu.push_back(*(std::string*)_chords.getFirstToken());
    _chords.release(1);
  }

  if (!shouldStop()) return PASS;

  // make sure we have one value for key and scale
  if (!_key.acquire(1) || !_scale.acquire(1)) {
    return NO_INPUT;
  }

  standard::Algorithm* algo = _chordsAlgo;
  string key = *(string*)_key.getFirstToken();
  string scale = *(string*)_scale.getFirstToken();
  vector<Real> chordsHist;
  Real cnr, ccr;
  string ckey, cscale;

  algo->input("chords").set(_accu);
  algo->input("key").set(key);
  algo->input("scale").set(scale);
  algo->output("chordsHistogram").set(chordsHist);
  algo->output("chordsNumberRate").set(cnr);
  algo->output("chordsChangesRate").set(ccr);
  algo->output("chordsKey").set(ckey);
  algo->output("chordsScale").set(cscale);
  algo->compute();

  _chordsHistogram.push(chordsHist);
  _chordsNumberRate.push(cnr);
  _chordsChangesRate.push(ccr);
  _chordsKey.push(ckey);
  _chordsScale.push(cscale);

  return FINISHED;
}

void ChordsDescriptors::reset() {
  Algorithm::reset();
  _accu.clear();
  if (_chordsAlgo) _chordsAlgo->reset();
}


} // namespace streaming
} // namespace essentia
