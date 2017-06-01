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

#include "keyExtended.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* KeyExtended::name = "KeyExtended";
const char* KeyExtended::category = "Tonal";
const char* KeyExtended::description = DOC("Using pitch profile classes, this algorithm calculates the best matching key estimate for a given HPCP. The algorithm was severely adapted and changed from the original implementation for readability and speed.\n"
"\n"
"Key will throw exceptions either when the input pcp size is not a positive multiple of 12 or if the key could not be found."
"\n"
"References:\n"
"  [1] E. Gómez, \"Tonal Description of Polyphonic Audio for Music Content\n"
"  Processing,\" INFORMS Journal on Computing, vol. 18, no. 3, pp. 294–304,\n"
"  2006.\n\n"
"  [2] Á. Faraldo, E. Gómez, S. Jordà, P.Herrera, \"Key Estimation in Electronic\n"
"  Dance Music. Proceedings of the 38th International Conference on information\n"
"  Retrieval, Padova, 2016.");


void KeyExtended::configure() {

  const char* keyNames[] = { "A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab" };
  _keys = arrayToVector<string>(keyNames);

  Real profileTypes[][12] = {

//  I     bII   II    bIII  III   IV    #IV   V     bVI   VI    bVII  VII
  { 1.00, 0.10, 0.43, 0.14, 0.61, 0.38, 0.12, 0.78, 0.13, 0.46, 0.15, 0.60 }, // ionian
  { 1.00, 0.10, 0.36, 0.37, 0.22, 0.33, 0.18, 0.75, 0.25, 0.18, 0.37, 0.37 }, // harmonic

  { 1.00, 0.10, 0.42, 0.10, 0.55, 0.40, 0.10, 0.77, 0.10, 0.42, 0.66, 0.15 }, // mixolydian
  { 1.00, 0.47, 0.10, 0.36, 0.24, 0.37, 0.16, 0.76, 0.30, 0.20, 0.45, 0.23 }, // phrygian
  
  { 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.65, 0.00, 0.00, 0.00, 0.00 }, // fifth
  { 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 }, // monotonic
  
  { 0.80, 0.60, 0.80, 0.60, 0.80, 0.60, 0.80, 0.60, 0.80, 0.60, 0.80, 0.60 }, // difficult
  { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 }, // empty

  { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 }, // empty
  { 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00 }, // empty
//  I     bII   II    bIII  III   IV    #IV   V     bVI   VI    bVII  VII    
  
};


#define SET_PROFILE(i) _M1 = arrayToVector<Real>(profileTypes[10*i]); _m1 = arrayToVector<Real>(profileTypes[10*i+1]); _M2 = arrayToVector<Real>(profileTypes[10*i+2]); _m2 = arrayToVector<Real>(profileTypes[10*i+3]); _M3 = arrayToVector<Real>(profileTypes[10*i+4]); _m3 = arrayToVector<Real>(profileTypes[10*i+5]); _M4 = arrayToVector<Real>(profileTypes[10*i+6]); _m4 = arrayToVector<Real>(profileTypes[10*i+7]); _P = arrayToVector<Real>(profileTypes[10*i+8]); _F = arrayToVector<Real>(profileTypes[10*i+9])

  SET_PROFILE(0);
  resize(parameter("pcpSize").toInt());
}


void KeyExtended::compute() {

  const vector<Real>& pcp = _pcp.get();

  int pcpsize = (int)pcp.size();
  int n = pcpsize/12;

  if (pcpsize < 12 || pcpsize % 12 != 0)
    throw EssentiaException("KeyExtended: input PCP size is not a positive multiple of 12");

  if (pcpsize != (int)_profile_dom1.size()) {
    resize(pcpsize);
  }

  // Compute Correlation
  // Means
  Real mean_pcp = mean(pcp);
  Real std_pcp = 0;

  // Standard Deviations
  for (int i=0; i<pcpsize; i++)
    std_pcp += (pcp[i] - mean_pcp) * (pcp[i] - mean_pcp);
  std_pcp = sqrt(std_pcp);

  // Correlation Matrix
  int keyIndex = -1;     // index of the first maximum
  Real max     = -1;     // first maximum
  Real max2    = -1;     // second maximum
  int scale    = MAJOR1; // scale

  // Compute maximum for all profiles.
  Real maxMaj1     = -1;
  Real max2Maj1    = -1;
  int keyIndexMaj1 = -1;

  Real maxMin1     = -1;
  Real max2Min1    = -1;
  int keyIndexMin1 = -1;

  Real maxMaj2     = -1;
  Real max2Maj2    = -1;
  int keyIndexMaj2 = -1;

  Real maxMin2     = -1;
  Real max2Min2    = -1;
  int keyIndexMin2 = -1;

  Real maxMaj3     = -1;
  Real max2Maj3    = -1;
  int keyIndexMaj3 = -1;

  Real maxMin3     = -1;
  Real max2Min3    = -1;
  int keyIndexMin3 = -1;

  Real maxMaj4     = -1;
  Real max2Maj4    = -1;
  int keyIndexMaj4 = -1;

  Real maxMin4     = -1;
  Real max2Min4    = -1;
  int keyIndexMin4 = -1;

  Real maxPeak     = -1;
  Real max2Peak    = -1;
  int keyIndexPeak = -1;

  Real maxFlat     = -1;
  Real max2Flat    = -1;
  int keyIndexFlat = -1;


  // calculate the correlation between the profiles and the PCP...
  // we shift the profile around to find the best match
  for (int shift=0; shift<pcpsize; shift++) {
    Real corrMaj1 = correlation(pcp, mean_pcp, std_pcp, _profile_doM1, _mean_profile_M1, _std_profile_M1, shift);
    if (corrMaj1 > maxMaj1) {
      max2Maj1 = maxMaj1;
      maxMaj1 = corrMaj1;
      keyIndexMaj1 = shift;
    }

    Real corrMin1 = correlation(pcp, mean_pcp, std_pcp, _profile_dom1, _mean_profile_m1, _std_profile_m1, shift);
    if (corrMin1 > maxMin1) {
      max2Min1 = maxMin1;
      maxMin1 = corrMin1;
      keyIndexMin1 = shift;
    }

    Real corrMaj2 = correlation(pcp, mean_pcp, std_pcp, _profile_doM2, _mean_profile_M2, _std_profile_M2, shift);
    if (corrMaj2 > maxMaj2) {
      max2Maj2 = maxMaj2;
      maxMaj2 = corrMaj2;
      keyIndexMaj2 = shift;
    }

    Real corrMin2 = correlation(pcp, mean_pcp, std_pcp, _profile_dom2, _mean_profile_m2, _std_profile_m2, shift);
    if (corrMin2 > maxMin2) {
      max2Min2 = maxMin2;
      maxMin2 = corrMin2;
      keyIndexMin2 = shift;
    }   

    Real corrMaj3 = correlation(pcp, mean_pcp, std_pcp, _profile_doM3, _mean_profile_M3, _std_profile_M3, shift);
    if (corrMaj3 > maxMaj3) {
      max2Maj3 = maxMaj3;
      maxMaj3 = corrMaj3;
      keyIndexMaj3 = shift;
    }

    Real corrMin3 = correlation(pcp, mean_pcp, std_pcp, _profile_dom3, _mean_profile_m3, _std_profile_m3, shift);
    if (corrMin3 > maxMin3) {
      max2Min3 = maxMin3;
      maxMin3 = corrMin3;
      keyIndexMin3 = shift;
    }  

    Real corrMaj4 = correlation(pcp, mean_pcp, std_pcp, _profile_doM4, _mean_profile_M4, _std_profile_M4, shift);
    if (corrMaj4 > maxMaj4) {
      max2Maj4 = maxMaj4;
      maxMaj4 = corrMaj4;
      keyIndexMaj4 = shift;
    }

    Real corrMin4 = correlation(pcp, mean_pcp, std_pcp, _profile_dom4, _mean_profile_m4, _std_profile_m4, shift);
    if (corrMin4 > maxMin4) {
      max2Min4 = maxMin4;
      maxMin4 = corrMin4;
      keyIndexMin4 = shift;
    }  

    Real corrPeak = correlation(pcp, mean_pcp, std_pcp, _profile_doP, _mean_profile_P, _std_profile_P, shift);
    if (corrPeak > maxPeak) {
      max2Peak = maxPeak;
      maxPeak = corrPeak;
      keyIndexPeak = shift;
    }

	Real corrFlat = correlation(pcp, mean_pcp, std_pcp, _profile_doF, _mean_profile_F, _std_profile_F, shift);
    if (corrFlat > maxFlat) {
      max2Flat = maxFlat;
      maxFlat = corrFlat;
      keyIndexFlat = shift;
    }
  }


  if (maxMaj1 > maxMin1 && maxMaj1 > maxMaj2 && maxMaj1 > maxMin2 && maxMaj1 > maxMaj3 && maxMaj1 > maxMin3 && maxMaj1 > maxMaj4 && maxMaj1 > maxMin4 && maxMaj1 > maxPeak && maxMaj1 > maxFlat) {
    keyIndex = (int) (keyIndexMaj1 *  12 / pcpsize + 0.5);
    scale = MAJOR1;
    max = maxMaj1;
    max2 = max2Maj1;
  }

  else if (maxMin1 > maxMaj1 && maxMin1 > maxMaj2 && maxMin1 > maxMin2 && maxMin1 > maxMaj3 && maxMin1 > maxMin3 && maxMin1 > maxMaj4 && maxMin1 > maxMin4 && maxMin1 > maxPeak && maxMin1 > maxFlat) {
    keyIndex = (int) (keyIndexMin1 * 12 / pcpsize + 0.5);
    scale = MINOR1;
    max = maxMin1;
    max2 = max2Min1;
    }

  else if (maxMaj2 > maxMaj1 && maxMaj2 > maxMin1 && maxMaj2 > maxMin2 && maxMaj2 > maxMaj3 && maxMaj2 > maxMin3 && maxMaj2 > maxMaj4 && maxMaj2 > maxMin4 && maxMaj2 > maxPeak && maxMaj2 > maxFlat) {
    keyIndex = (int) (keyIndexMaj2 * 12 / pcpsize + 0.5);
    scale = MAJOR2;
    max = maxMaj2;
    max2 = max2Maj2;
    }
  
  else if (maxMin2 > maxMaj1 && maxMin2 > maxMin1 && maxMin2 > maxMaj2 && maxMin2 > maxMaj3 && maxMin2 > maxMin3 && maxMin2 > maxMaj4 && maxMin2 > maxMin4 && maxMin2 > maxPeak && maxMin2 > maxFlat) {
    keyIndex = (int) (keyIndexMin2 * 12 / pcpsize + 0.5);
    scale = MINOR2;
    max = maxMin2;
    max2 = max2Min2;
    }

  else if (maxMaj3 > maxMaj1 && maxMaj3 > maxMin1 && maxMaj3 > maxMaj2 && maxMaj3 > maxMin2 && maxMaj3 > maxMin3 && maxMaj3 > maxMaj4 && maxMaj3 > maxMin4 && maxMaj3 > maxPeak && maxMaj3 > maxFlat) {
    keyIndex = (int) (keyIndexMaj3 * 12 / pcpsize + 0.5);
    scale = MAJOR3;
    max = maxMaj3;
    max2 = max2Maj3;
    }
  
  else if (maxMin3 > maxMaj1 && maxMin3 > maxMin1 && maxMin3 > maxMaj2 && maxMin3 > maxMin2 && maxMin3 > maxMaj3 && maxMin3 > maxMaj4 && maxMin3 > maxMin4 && maxMin3 > maxPeak && maxMin3 > maxFlat) {
    keyIndex = (int) (keyIndexMin3 * 12 / pcpsize + 0.5);
    scale = MINOR3;
    max = maxMin3;
    max2 = max2Min3;
    }

  else if (maxMaj4 > maxMaj1 && maxMaj4 > maxMin1 && maxMaj4 > maxMaj2 && maxMaj4 > maxMin2 && maxMaj4 > maxMaj3 && maxMaj4 > maxMin3 && maxMaj4 > maxMin4 && maxMaj4 > maxPeak && maxMaj4 > maxFlat) {
    keyIndex = (int) (keyIndexMaj4 * 12 / pcpsize + 0.5);
    scale = MAJOR4;
    max = maxMaj4;
    max2 = max2Maj4;
    }
  
  else if (maxMin4 > maxMaj1 && maxMin4 > maxMin1 && maxMin4 > maxMaj2 && maxMin4 > maxMin2 && maxMin4 > maxMaj3 && maxMin4 > maxMin3 && maxMin4 > maxMaj4 && maxMin4 > maxPeak && maxMin4 > maxFlat) {
    keyIndex = (int) (keyIndexMin4 * 12 / pcpsize + 0.5);
    scale = MINOR4;
    max = maxMin4;
    max2 = max2Min4;
    }

  else if (maxPeak > maxMaj1 && maxPeak > maxMin1 && maxPeak > maxMaj2 && maxPeak > maxMin2 && maxPeak > maxMaj3 && maxPeak > maxMin3 && maxPeak > maxMaj4 && maxPeak > maxMin4 && maxPeak > maxFlat) {
    keyIndex = (int) (keyIndexPeak * 12 / pcpsize + 0.5);
    scale = PEAK;
    max = maxPeak;
    max2 = max2Peak;
    }

  else {
    keyIndex = (int) (keyIndexFlat * 12 / pcpsize + 0.5);
    scale = FLAT;
    max = maxFlat;
    max2 = max2Flat;
  }

  if (keyIndex < 0) {
    throw EssentiaException("KeyExtended: keyIndex smaller than zero. Could not find key.");
  }

  //////////////////////////////////////////////////////////////////////////////
  // Here we calculate the outputs...

  // first three outputs are key, scale and strength
  _key.get() = _keys[keyIndex];

  if (scale == MAJOR1) {
    _scale.get() = "ionian";
  }

  else if (scale == MINOR1) {
    _scale.get() = "harmonic";
  }

  else if (scale == MAJOR2) {
    _scale.get() = "mixolydian";
  }

  else if (scale == MINOR2) {
    _scale.get() = "phrygian";
  }

  else if (scale == MAJOR3) {
    _scale.get() = "fifth";
  }

  else if (scale == MINOR3) {
    _scale.get() = "monotonic";
  }

  else if (scale == MAJOR4) {
    _scale.get() = "difficult";
  }

  else if (scale == MINOR4) {
    _scale.get() = "empty";
  }

  else if (scale == PEAK) {
    _scale.get() = "empty";
  }

  else if (scale == FLAT) {
    _scale.get() = "empty";
  }

	else {
    _scale.get() = "unknown";
  }     

  _strength.get() = max;

  // this one outputs the relative difference between the maximum and the
  // second highest maximum (i.e. Compute second highest correlation peak)
  _firstToSecondRelativeStrength.get() = (max - max2) / max;
}


// this function resizes and interpolates the profiles to fit the
// pcp size...
void KeyExtended::resize(int pcpsize) {
  ///////////////////////////////////////////////////////////////////
  // Interpolate to get pcpsize values
  int n = pcpsize/12;

  _profile_doM1.resize(pcpsize);
  _profile_dom1.resize(pcpsize);
  _profile_doM2.resize(pcpsize);
  _profile_dom2.resize(pcpsize);
  _profile_doM3.resize(pcpsize);
  _profile_dom3.resize(pcpsize);
  _profile_doM4.resize(pcpsize);
  _profile_dom4.resize(pcpsize);
  _profile_doP.resize(pcpsize);
  _profile_doF.resize(pcpsize);

  for (int i=0; i<12; i++) {
    _profile_doM1[i*n] = _M1[i];
    _profile_dom1[i*n] = _m1[i];
    _profile_doM2[i*n] = _M2[i];
    _profile_dom2[i*n] = _m2[i];
    _profile_doM3[i*n] = _M3[i];
    _profile_dom3[i*n] = _m3[i];
    _profile_doM4[i*n] = _M4[i];
    _profile_dom4[i*n] = _m4[i];
    _profile_doP[i*n]  = _P[i];
    _profile_doF[i*n]  = _F[i];

    // Two interpolated values
    Real incr_M1, incr_m1, incr_M2, incr_m2, incr_M3, incr_m3, incr_M4, incr_m4, incr_P, incr_F;
    if (i == 11) {
      incr_M1 = (_M1[11] - _M1[0]) / n;
      incr_m1 = (_m1[11] - _m1[0]) / n;
      incr_M2 = (_M2[11] - _M2[0]) / n;
      incr_m2 = (_m2[11] - _m2[0]) / n;
      incr_M3 = (_M3[11] - _M3[0]) / n;
      incr_m3 = (_m3[11] - _m3[0]) / n;
      incr_M4 = (_M4[11] - _M4[0]) / n;
      incr_m4 = (_m4[11] - _m4[0]) / n;
      incr_P = (_P[11] - _P[0]) / n;
      incr_F = (_F[11] - _F[0]) / n;
    }
    else {
      incr_M1 = (_M1[i] - _M1[i+1]) / n;
      incr_m1 = (_m1[i] - _m1[i+1]) / n;
      incr_M2 = (_M2[i] - _M2[i+1]) / n;
      incr_m2 = (_m2[i] - _m2[i+1]) / n;
      incr_M3 = (_M3[i] - _M3[i+1]) / n;
      incr_m3 = (_m3[i] - _m3[i+1]) / n;
      incr_M4 = (_M4[i] - _M4[i+1]) / n;
      incr_m4 = (_m4[i] - _m4[i+1]) / n;
      incr_P = (_P[i] - _P[i+1]) / n;
      incr_F = (_F[i] - _F[i+1]) / n;
    }

    for (int j=1; j<=(n-1); j++) {
      _profile_doM1[i*n+j] = _M1[i] - j * incr_M1;
      _profile_dom1[i*n+j] = _m1[i] - j * incr_m1;
      _profile_doM2[i*n+j] = _M2[i] - j * incr_M2;
      _profile_dom2[i*n+j] = _m2[i] - j * incr_m2;
      _profile_doM3[i*n+j] = _M3[i] - j * incr_M3;
      _profile_dom3[i*n+j] = _m3[i] - j * incr_m3;
      _profile_doM4[i*n+j] = _M4[i] - j * incr_M4;
      _profile_dom4[i*n+j] = _m4[i] - j * incr_m4;
     	_profile_doP[i*n+j] = _P[i] - j * incr_P;
      _profile_doF[i*n+j] = _F[i] - j * incr_F;			
    }
  }

  _mean_profile_M1 = mean(_profile_doM1);
  _mean_profile_m1 = mean(_profile_dom1);
  _mean_profile_M2 = mean(_profile_doM2);
  _mean_profile_m2 = mean(_profile_dom2);
  _mean_profile_M3 = mean(_profile_doM3);
  _mean_profile_m3 = mean(_profile_dom3);
  _mean_profile_M4 = mean(_profile_doM4);
  _mean_profile_m4 = mean(_profile_dom4);
  _mean_profile_P  = mean(_profile_doP);
  _mean_profile_F  = mean(_profile_doF);

  _std_profile_M1 = 0;
  _std_profile_m1 = 0;
  _std_profile_M2 = 0;
  _std_profile_m2 = 0;
  _std_profile_M3 = 0;
  _std_profile_m3 = 0;
  _std_profile_M4 = 0;
  _std_profile_m4 = 0;
  _std_profile_P  = 0;
  _std_profile_F  = 0;

  // Compute Standard Deviations
  for (int i=0; i<pcpsize; i++) {
    _std_profile_M1 += (_profile_doM1[i] - _mean_profile_M1) * (_profile_doM1[i] - _mean_profile_M1);
    _std_profile_m1 += (_profile_dom1[i] - _mean_profile_m1) * (_profile_dom1[i] - _mean_profile_m1);
    _std_profile_M2 += (_profile_doM2[i] - _mean_profile_M2) * (_profile_doM2[i] - _mean_profile_M2);
    _std_profile_m2 += (_profile_dom2[i] - _mean_profile_m2) * (_profile_dom2[i] - _mean_profile_m2);
    _std_profile_M3 += (_profile_doM3[i] - _mean_profile_M3) * (_profile_doM3[i] - _mean_profile_M3);
    _std_profile_m3 += (_profile_dom3[i] - _mean_profile_m3) * (_profile_dom3[i] - _mean_profile_m3);
    _std_profile_M4 += (_profile_doM4[i] - _mean_profile_M4) * (_profile_doM4[i] - _mean_profile_M4);
    _std_profile_m4 += (_profile_dom4[i] - _mean_profile_m4) * (_profile_dom4[i] - _mean_profile_m4);
    _std_profile_P  += (_profile_doP[i]  - _mean_profile_P)  * (_profile_doP[i]  - _mean_profile_P);
    _std_profile_F  += (_profile_doF[i]  - _mean_profile_F)  * (_profile_doF[i]  - _mean_profile_F);
  }
  _std_profile_M1 = sqrt(_std_profile_M1);
  _std_profile_m1 = sqrt(_std_profile_m1);
  _std_profile_M2 = sqrt(_std_profile_M2);
  _std_profile_m2 = sqrt(_std_profile_m2);
  _std_profile_M3 = sqrt(_std_profile_M3);
  _std_profile_m3 = sqrt(_std_profile_m3);
  _std_profile_M4 = sqrt(_std_profile_M4);
  _std_profile_m4 = sqrt(_std_profile_m4);
  _std_profile_P  = sqrt(_std_profile_P);
  _std_profile_F  = sqrt(_std_profile_F);
}


// correlation coefficient with 'shift'
// one of the vectors is shifted in time, and then the correlation is calculated,
// just like a cross-correlation
Real KeyExtended::correlation(const vector<Real>& v1, const Real mean1, const Real std1, const vector<Real>& v2, const Real mean2, const Real std2, const int shift) const
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


} // namespace standard
} // namespace essentia

#include "poolstorage.h"
#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

const char* KeyExtended::name = standard::KeyExtended::name;
const char* KeyExtended::category = standard::KeyExtended::category;
const char* KeyExtended::description = standard::KeyExtended::description;

KeyExtended::KeyExtended() : AlgorithmComposite() {

  _keyExtendedAlgo = standard::AlgorithmFactory::create("KeyExtended");
  _poolStorage = new PoolStorage<std::vector<Real> >(&_pool, "internal.hpcp");

  declareInput(_poolStorage->input("data"), 1, "pcp", "the input pitch class profile");

  declareOutput(_key, 0, "key", "the estimated key, from A to G");
  declareOutput(_scale, 0, "scale", "the scale of the key (major, minor or unknown)");
  declareOutput(_strength, 0, "strength", "the strength of the estimated key");
}

KeyExtended::~KeyExtended() {
  delete _keyExtendedAlgo;
  delete _poolStorage;
}


AlgorithmStatus KeyExtended::process() {
  if (!shouldStop()) return PASS;

  const vector<vector<Real> >& hpcpKey = _pool.value<vector<vector<Real> > >("internal.hpcp");
  vector<Real> hpcpAverage = meanFrames(hpcpKey);
  string key;
  string scale;
  Real strength;
  Real firstToSecondRelativeStrength;
  _keyExtendedAlgo->input("pcp").set(hpcpAverage);
  _keyExtendedAlgo->output("key").set(key);
  _keyExtendedAlgo->output("scale").set(scale);
  _keyExtendedAlgo->output("strength").set(strength);
  _keyExtendedAlgo->output("firstToSecondRelativeStrength").set(firstToSecondRelativeStrength);
  _keyExtendedAlgo->compute();

  _key.push(key);
  _scale.push(scale);
  _strength.push(strength);

  return FINISHED;
}


void KeyExtended::reset() {
  AlgorithmComposite::reset();
  _keyExtendedAlgo->reset();
}

} // namespace streaming
} // namespace essentia
