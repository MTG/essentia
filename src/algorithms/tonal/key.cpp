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

#include "key.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* Key::name = "Key";
const char* Key::category = "Tonal";
const char* Key::description = DOC("This algorithm computes key estimate given a pitch class profile (HPCP). The algorithm was severely adapted and changed from the original implementation for readability and speed.\n"
"\n"
"Key will throw exceptions either when the input pcp size is not a positive multiple of 12 or if the key could not be found. Also if parameter \"scale\" is set to \"minor\" and the profile type is set to \"weichai\"\n"
"\n"
"  Abouth the Key Profiles:\n"
"  - 'Diatonic' - binary profile with diatonic notes of both modes. Could be useful for ambient music or diatonic music which is not strictly 'tonal functional'.\n"
"  - 'Tonic Triad' - just the notes of the major and minor chords. Exclusively for testing.\n"
"  - 'Krumhansl' - reference key profiles after cognitive experiments with users. They should work generally fine for pop music.\n"
"  - 'Temperley' - key profiles extracted from corpus analysis of euroclassical music. Therefore, they perform best on this repertoire (especially in minor).\n"
"  - 'Shaath' -  profiles based on Krumhansl's specifically tuned to popular and electronic music.\n"
"  - 'Noland' - profiles from Bach's 'Well Tempered Klavier'.\n" 
"  - 'edma' - automatic profiles extracted from corpus analysis of electronic dance music [3]. They normally perform better that Shaath's\n"
"  - 'edmm' - automatic profiles extracted from corpus analysis of electronic dance music and manually tweaked according to heuristic observation. It will report major modes (which are poorly represented in EDM) as minor, but improve performance otherwise [3].\n"
"  - Other key profiles ('Faraldo', 'Pentatonic') are experimental and will be removed on due time.\n"

"References:\n"
"  [1] E. Gómez, \"Tonal Description of Polyphonic Audio for Music Content\n"
"  Processing,\" INFORMS Journal on Computing, vol. 18, no. 3, pp. 294–304,\n"
"  2006.\n\n"
"  [2] D. Temperley, \"What's key for key? The Krumhansl-Schmuckler\n"
"  key-finding algorithm reconsidered\", Music Perception vol. 17, no. 1,\n"
"  pp. 65-100, 1999."
"  [3] Á. Faraldo, E. Gómez, S. Jordà, P.Herrera, \"Key Estimation in Electronic\n"
"  Dance Music. Proceedings of the 38th International Conference on information\n"
"  Retrieval, Padova, 2016. (In Press.)");


void Key::configure() {
  _slope = parameter("slope").toReal();
  _numHarmonics = parameter("numHarmonics").toInt();
  _profileType = parameter("profileType").toString();

  const char* keyNames[] = { "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#" };
  _keys = arrayToVector<string>(keyNames);

  Real profileTypes[][12] = {
    // Diatonic
    { 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1 },
    { 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1 },

    // Krumhansl
    { 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88 },
    { 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17 },

    // A revised version of the key profiles, by David Temperley, see [2]
    { 5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0 },
    { 5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0 },

    // Wei Chai MIT PhD thesis
    { 81302, 320, 65719, 1916, 77469, 40928, 2223, 83997, 1218, 39853, 1579, 28908 },
    { 39853, 1579, 28908, 81302, 320, 65719, 1916, 77469, 40928, 2223, 83997, 1218 },

    // Tonic triad.
    { 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0 },
    { 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0 },

    // Temperley MIREX 2005
    { 0.748, 0.060, 0.488, 0.082, 0.67, 0.46, 0.096, 0.715, 0.104, 0.366, 0.057, 0.4 },
    { 0.712, 0.084, 0.474, 0.618, 0.049, 0.46, 0.105, 0.747, 0.404, 0.067, 0.133, 0.33 },

    // Statistics THPCP over all the evaluation set
    { 0.95162, 0.20742, 0.71758, 0.22007, 0.71341, 0.48841, 0.31431, 1.00000, 0.20957, 0.53657, 0.22585, 0.55363 },
    { 0.94409, 0.21742, 0.64525, 0.63229, 0.27897, 0.57709, 0.26428, 1.0000, 0.26428, 0.30633, 0.45924, 0.35929 },

    // Shaath
    { 6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4 },
    { 6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2 },

    // Gómez (as specified by Shaath)
    { 0.82, 0.00, 0.55, 0.00, 0.53, 0.30, 0.08, 1.00, 0.00, 0.38, 0.00, 0.47 },
    { 0.81, 0.00, 0.53, 0.54, 0.00, 0.27, 0.07, 1.00, 0.27, 0.07, 0.10, 0.36 },

    // Noland
    { 0.0629, 0.0146, 0.061, 0.0121, 0.0623, 0.0414, 0.0248, 0.0631, 0.015, 0.0521, 0.0142, 0.0478 },
    { 0.0682, 0.0138, 0.0543, 0.0519, 0.0234, 0.0544, 0.0176, 0.067, 0.0349, 0.0297, 0.0401, 0.027 },

    // Faraldo
    { 7.0, 2.0, 3.8, 2.3, 4.7, 4.1, 2.5, 5.2, 2.0, 3.7, 3.0, 3.4 },
    { 7.0, 3.0, 3.8, 4.5, 2.6, 3.5, 2.5, 5.2, 4.0, 2.5, 4.5, 3.0 },

    // Pentatonic
    { 1.0, 0.1, 0.25, 0.1, 0.5, 0.7, 0.1, 0.8, 0.1, 0.25, 0.1, 0.5 },
    { 1.0, 0.2, 0.25, 0.5, 0.1, 0.7, 0.1, 0.8, 0.3, 0.2, 0.6, 0.2  },

    // edmm
    { 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083, 0.083 },
    { 0.17235348, 0.04, 0.0761009,  0.12, 0.05621498, 0.08527853, 0.0497915,  0.13451001, 0.07458916, 0.05003023, 0.09187879, 0.05545106 },

    // edma
    { 0.16519551, 0.04749026, 0.08293076, 0.06687112, 0.09994645, 0.09274123, 0.05294487, 0.13159476, 0.05218986, 0.07443653, 0.06940723, 0.0642515  },
    { 0.17235348, 0.05336489, 0.0761009,  0.10043649, 0.05621498, 0.08527853, 0.0497915,  0.13451001, 0.07458916, 0.05003023, 0.09187879, 0.05545106 }
};


#define SET_PROFILE(i) _M = arrayToVector<Real>(profileTypes[2*i]); _m = arrayToVector<Real>(profileTypes[2*i+1])

  if      (_profileType == "diatonic")      { SET_PROFILE(0);  }
  else if (_profileType == "krumhansl")     { SET_PROFILE(1);  }
  else if (_profileType == "temperley")     { SET_PROFILE(2);  }
  else if (_profileType == "weichai")       { SET_PROFILE(3);  }
  else if (_profileType == "tonictriad")    { SET_PROFILE(4);  }
  else if (_profileType == "temperley2005") { SET_PROFILE(5);  }
  else if (_profileType == "thpcp")         { SET_PROFILE(6);  }
  else if (_profileType == "shaath")        { SET_PROFILE(7);  }
  else if (_profileType == "gomez")         { SET_PROFILE(8);  }
  else if (_profileType == "noland")        { SET_PROFILE(9);  }
  else if (_profileType == "faraldo")       { SET_PROFILE(10); }
  else if (_profileType == "pentatonic")    { SET_PROFILE(11); }
  else if (_profileType == "edmm")          { SET_PROFILE(12); }
  else if (_profileType == "edma")          { SET_PROFILE(13); }
  else {
    throw EssentiaException("Key: Unsupported profile type: ", _profileType);
  }

  // Compute the other vectors getting into account chords:
  vector<Real> M_chords(12, (Real)0.0);
  vector<Real> m_chords(12, (Real)0.0);

  /* Under test: Purwins et al.
  for (int n=0; n<12; n++) {
    TIndex dominant = n+7;
    if ( dominant > 11)
      dominant -= 12;
    M_chords[n]= _M[n] + (1.0/3.0)*_M[dominant];
    m_chords[n]= _m[n] + (1.0/3.0)*_m[dominant];
  }
  */

  /*
  Assumptions:
    - We consider that the tonal hierarchy is kept when dealing with polyphonic sounds.
      That means that Krumhansl profiles are seen as the tonal hierarchy of
      each of the chords of the harmonic scale within a major/minor tonal contest.
    - We compute from these chord profiles the corresponding note (pitch class) profiles,
      which will be compared to HPCP values.

  Rationale:
    - Each note contribute to the different harmonics.
    - All the chords of the major/minor key are considered.

  Procedure:
    - First, profiles are initialized to 0
    - We take _M[i], n[i] as Krumhansl profiles i=1,...12 related to each of the chords
      of the major/minor key.
    - For each chord, we add its contribution to the three notes (pitch classes) of the chord.
      We use the same weight for all the notes of the chord.
    - For each note, we add its contribution to the different harmonics
  */

  /** MAJOR KEY */
  // Tonic (I)
  addMajorTriad(0, _M[0], M_chords);

  if (!parameter("useThreeChords").toBool())
  {
    // II
    addMinorTriad(2, _M[2], M_chords);
    // Only root: AddContributionHarmonics(2, _M[2], M_chords);
    // III
    addMinorTriad(4, _M[4], M_chords);
    // Only root: AddContributionHarmonics(4, _M[4], M_chords);
  }

  // Subdominant (IV)
  addMajorTriad(5, _M[5], M_chords);
  // Dominant (V)
  addMajorTriad(7, _M[7], M_chords);

  if (!parameter("useThreeChords").toBool()) {
    // VI
    addMinorTriad(9, _M[9], M_chords);
    // Only root: AddContributionHarmonics(9, _M[9], M_chords);
    // VII (5th diminished)
    addContributionHarmonics(11, _M[11], M_chords);
    addContributionHarmonics(2 , _M[11], M_chords);
    addContributionHarmonics(5 , _M[11], M_chords);
    // Only root: AddContributionHarmonics(11, _M[11], M_chords);
  }

  /** MINOR KEY */
  // Tonica I
  addMinorTriad(0, _m[0], m_chords);
  if (!parameter("useThreeChords").toBool()){
    // II (5th diminished)
    addContributionHarmonics(2, _m[2], m_chords);
    addContributionHarmonics(5, _m[2], m_chords);
    addContributionHarmonics(8, _m[2], m_chords);
    // Only root: AddContributionHarmonics(2, _m[2], m_chords);

    // III (5th augmented)
    addContributionHarmonics(3, _m[3], m_chords);
    addContributionHarmonics(7, _m[3], m_chords);
    addContributionHarmonics(11,_m[3], m_chords); // Harmonic minor scale! antes 10!!!
    // Only root: AddContributionHarmonics(3, _m[3], m_chords);
  }

  // Subdominant (IV)
  addMinorTriad(5, _m[5], m_chords);

  // Dominant (V) (harmonic minor scale)
  addMajorTriad(7, _m[7], m_chords);

  if (!parameter("useThreeChords").toBool()) {
    // VI
    addMajorTriad(8, _m[8], m_chords);
    // Only root: AddContributionHarmonics(8, _m[8], m_chords);
    // VII (diminished 5th)
    addContributionHarmonics(11, _m[8], m_chords);
    addContributionHarmonics(2, _m[8], m_chords);
    addContributionHarmonics(5, _m[8], m_chords);
    // Only root: AddContributionHarmonics(11, _m[8], m_chords);
  }

  if (parameter("usePolyphony").toBool()) {
    _M = M_chords;
    _m = m_chords;
  }

  resize(parameter("pcpSize").toInt());
}


void Key::compute() {

  const vector<Real>& pcp = _pcp.get();

  int pcpsize = (int)pcp.size();
  int n = pcpsize/12;

  if (pcpsize < 12 || pcpsize % 12 != 0)
    throw EssentiaException("Key: input PCP size is not a positive multiple of 12");

  if (pcpsize != (int)_profile_dom.size()) {
    resize(pcpsize);
  }

  ///////////////////////////////////////////////////////////////////
  // compute correlation

  // Compute means
  Real mean_pcp = mean(pcp);
  Real std_pcp = 0;

  // Compute Standard Deviations
  for (int i=0; i<pcpsize; i++)
    std_pcp += (pcp[i] - mean_pcp) * (pcp[i] - mean_pcp);
  std_pcp = sqrt(std_pcp);

  // Compute correlation matrix
  int keyIndex = -1; // index of the first maximum
  Real max = -1;     // first maximum
  Real max2 = -1;    // second maximum
  int scale = MAJOR;  // scale

  // Compute maximum for both major and minor
  Real maxMaj = -1;
  Real max2Maj = -1;
  int keyIndexMaj = -1;

  Real maxMin = -1;
  Real max2Min = -1;
  int keyIndexMin = -1;

  // calculate the correlation between the profiles and the PCP...
  // we shift the profile around to find the best match
  for (int shift=0; shift<pcpsize; shift++) {
    /*
    // Penalization if the Tonic has not a minimum amplitude
    // max_pcp needs to be calculated...
    Real factor = pcp[i]/max_pcp;
    if (factor < 0.6) {
      corrMajor *= factor / 0.6;
      corrMinor *= factor / 0.6;
    }
    */
    Real corrMajor = correlation(pcp, mean_pcp, std_pcp, _profile_doM, _mean_profile_M, _std_profile_M, shift);
    // Compute maximum value for major keys
    if (corrMajor > maxMaj) {
      max2Maj = maxMaj;
      maxMaj = corrMajor;
      keyIndexMaj = shift;
    }

    Real corrMinor = correlation(pcp, mean_pcp, std_pcp, _profile_dom, _mean_profile_m, _std_profile_m, shift);
    // Compute maximum value for minor keys
    if (corrMinor > maxMin) {
      max2Min = maxMin;
      maxMin = corrMinor;
      keyIndexMin = shift;
    }
  }

  if (maxMaj >= maxMin) {
    keyIndex = (int) (keyIndexMaj *  12 / pcpsize + .5);
    scale = MAJOR;
    max = maxMaj;
    max2 = max2Maj;
  }
  else {
    keyIndex = (int) (keyIndexMin * 12 / pcpsize + .5);
    scale = MINOR;
    max = maxMin;
    max2 = max2Min;
  }

  // In the case of Wei Chai algorithm, the scale is detected in a second step
  // In this point, always the major relative is detected, as it is the first
  // maximum
  if (_profileType == "weichai") {
    if (scale == MINOR)
      throw EssentiaException("Key: error in Wei Chai algorithm. Wei Chai algorithm does not support minor scales.");

    int fifth = keyIndex + 7*n;
    if (fifth > pcpsize)
      fifth -= pcpsize;
    int sixth = keyIndex + 9*n;
    if (sixth > pcpsize)
      sixth -= pcpsize;

    if (pcp[sixth] >  pcp[fifth]) {
      keyIndex = sixth;
      keyIndex = (int) (keyIndex * 12 / pcpsize + .5);
      scale = MINOR;
    }
  }

  // keyIndex = (int)(keyIndex * 12.0 / pcpsize + 0.5) % 12;

  if (keyIndex < 0) {
    throw EssentiaException("Key: keyIndex smaller than zero. Could not find key.");
  }

  //////////////////////////////////////////////////////////////////////////////
  // Here we calculate the outputs...

  // first three outputs are key, scale and strength
  _key.get() = _keys[keyIndex];
  _scale.get() = scale == MAJOR ? "major" : "minor";
  _strength.get() = max;

  // this one outputs the relative difference between the maximum and the
  // second highest maximum (i.e. Compute second highest correlation peak)
  _firstToSecondRelativeStrength.get() = (max - max2) / max;

}

// this function resizes and interpolates the profiles to fit the
// pcp size...
void Key::resize(int pcpsize) {
  ///////////////////////////////////////////////////////////////////
  // Interpolate to get pcpsize values
  int n = pcpsize/12;

  _profile_doM.resize(pcpsize);
  _profile_dom.resize(pcpsize);

  for (int i=0; i<12; i++) {

    _profile_doM[i*n] = _M[i];
    _profile_dom[i*n] = _m[i];

    // Two interpolated values
    Real incr_M, incr_m;
    if (i == 11) {
      incr_M = (_M[11] - _M[0]) / n;
      incr_m = (_m[11] - _m[0]) / n;
    }
    else {
      incr_M = (_M[i] - _M[i+1]) / n;
      incr_m = (_m[i] - _m[i+1]) / n;
    }

    for (int j=1; j<=(n-1); j++) {
      _profile_doM[i*n+j] = _M[i] - j * incr_M;
      _profile_dom[i*n+j] = _m[i] - j * incr_m;
    }
  }

  _mean_profile_M = mean(_profile_doM);
  _mean_profile_m = mean(_profile_dom);
  _std_profile_M = 0;
  _std_profile_m = 0;

  // Compute Standard Deviations
  for (int i=0; i<pcpsize; i++) {
    _std_profile_M += (_profile_doM[i] - _mean_profile_M) * (_profile_doM[i] - _mean_profile_M);
    _std_profile_m += (_profile_dom[i] - _mean_profile_m) * (_profile_dom[i] - _mean_profile_m);
  }
  _std_profile_M = sqrt(_std_profile_M);
  _std_profile_m = sqrt(_std_profile_m);
}


// correlation coefficient with 'shift'
// on of the vectors is shifted in time, and then the correlation is calculated,
// just like a cross-correlation
Real Key::correlation(const vector<Real>& v1, const Real mean1, const Real std1, const vector<Real>& v2, const Real mean2, const Real std2, const int shift) const
{
  Real r = 0.0;
  int size = (int)v1.size();

  for (int i=0; i<size; i++)
  {
    int index = (i - shift) % size;

    if (index < 0) {
      index += size;
    }

    r += (v1[i] - mean1) * (v2[index] - mean2);
  }

  r /= std1*std2;

  return r;
}

/**
  Each note contribute to the different harmonics:
  1.- first  harmonic  f   -> i
  2.- second harmonic  2*f -> i
  3.- third  harmonic  3*f -> i+7
  4.- fourth harmonic  4*f -> i
  ..
  The contribution is weighted depending of the slope
*/
void Key::addContributionHarmonics(const int pitchclass, const Real contribution, vector<Real>& M_chords) const
{
  Real weight = contribution;

  for (int iHarm = 1; iHarm <= _numHarmonics; iHarm++) {

    Real index  = pitchclass + 12*log2((Real)iHarm);

    Real before = floor(index);
    Real after  = ceil (index);

    int ibefore= (int) fmod((Real)before,(Real)12.0);
    int iafter = (int) fmod((Real)after ,(Real)12.0);

    // weight goes proportionally to ibefore & iafter
    if (ibefore < iafter) {
      Real distance_before = index-before;
      M_chords[ibefore] += pow(cos(0.5*M_PI*distance_before),2)*weight;

      Real distance_after  = after-index;
      M_chords[iafter ] += pow(cos(0.5*M_PI*distance_after ),2)*weight;
    }
    else { // equal
      M_chords[ibefore] += weight;
    }
    weight *= _slope;
  }
}

/**
  Function that adds the contribution of a chord with root note 'root' to its major triad
  A major triad includes notes from three different classes of pitch: the root, the major 3rd and perfect 5th.
  This is the most relaxed, most consonant chord in all of harmony.
  @see http://www.songtrellis.com/directory/1146/chordTypes/majorChordTypes/majorTriad
  The three notes of the chord have the same weight
*/
void Key::addMajorTriad(const int root, const Real contribution, vector<Real>& M_chords) const
{
  // Root
  addContributionHarmonics(root, contribution, M_chords);

  // Major 3rd
  int third = root + 4;
  if (third > 11)
    third -= 12;
  addContributionHarmonics(third, contribution, M_chords);

  // Perfect 5th
  int fifth = root + 7;
  if (fifth > 11)
    fifth -= 12;
  addContributionHarmonics(fifth, contribution, M_chords);

}
/**
  Function that adds the contribution of a chord with root note 'root' to its minor triad
  A minor triad includes notes from three different classes of pitch: the root, the minor 3rd and perfect 5th.
  @see http://www.songtrellis.com/directory/1146/chordTypes/minorChordTypes/minorTriadMi
  The three notes of the chord have the same weight
*/
void Key::addMinorTriad(int root, Real contribution, vector<Real>& M_chords) const
{
  // Root
  addContributionHarmonics(root, contribution, M_chords);

  // Minor 3rd
  int third = root+3;
  if (third > 11)
    third -= 12;
  addContributionHarmonics(third, contribution, M_chords);

  // Perfect 5th
  int fifth = root+7;
  if (fifth > 11)
    fifth -= 12;
  addContributionHarmonics(fifth, contribution, M_chords);
}

} // namespace standard
} // namespace essentia

#include "poolstorage.h"
#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

const char* Key::name = standard::Key::name;
const char* Key::category = standard::Key::category;
const char* Key::description = standard::Key::description;

Key::Key() : AlgorithmComposite() {

  _keyAlgo = standard::AlgorithmFactory::create("Key");
  _poolStorage = new PoolStorage<std::vector<Real> >(&_pool, "internal.hpcp");

  declareInput(_poolStorage->input("data"), 1, "pcp", "the input pitch class profile");
  declareOutput(_key, 0, "key", "the estimated key, from A to G");
  declareOutput(_scale, 0, "scale", "the scale of the key (major or minor)");
  declareOutput(_strength, 0, "strength", "the strength of the estimated key");
}

Key::~Key() {
  delete _keyAlgo;
  delete _poolStorage;
}


AlgorithmStatus Key::process() {
  if (!shouldStop()) return PASS;

  const vector<vector<Real> >& hpcpKey = _pool.value<vector<vector<Real> > >("internal.hpcp");
  vector<Real> hpcpAverage = meanFrames(hpcpKey);
  string key;
  string scale;
  Real strength;
  Real firstToSecondRelativeStrength;
  _keyAlgo->input("pcp").set(hpcpAverage);
  _keyAlgo->output("key").set(key);
  _keyAlgo->output("scale").set(scale);
  _keyAlgo->output("strength").set(strength);
  _keyAlgo->output("firstToSecondRelativeStrength").set(firstToSecondRelativeStrength);
  _keyAlgo->compute();

  _key.push(key);
  _scale.push(scale);
  _strength.push(strength);

  return FINISHED;
}


void Key::reset() {
  AlgorithmComposite::reset();
  _keyAlgo->reset();
}


} // namespace streaming
} // namespace essentia
