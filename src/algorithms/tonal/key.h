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

#ifndef ESSENTIA_KEY_H
#define ESSENTIA_KEY_H

#include "algorithm.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {

class Key : public Algorithm {

 private:
  Input<std::vector<Real> > _pcp;

  Output<std::string> _key;
  Output<std::string> _scale;
  Output<Real> _strength;
  Output<Real> _firstToSecondRelativeStrength;

 public:

  Key() {
    declareInput(_pcp, "pcp", "the input pitch class profile");
    declareOutput(_key, "key", "the estimated key, from A to G");
    declareOutput(_scale, "scale", "the scale of the key (major or minor)");
    declareOutput(_strength, "strength", "the strength of the estimated key");
    declareOutput(_firstToSecondRelativeStrength, "firstToSecondRelativeStrength", "the relative strength difference between the best estimate and second best estimate of the key");
  }

  void declareParameters() {
    declareParameter("usePolyphony", "enables the use of polyphonic profiles to define key profiles (this includes the contributions from triads as well as pitch harmonics)", "{true,false}", true);
    declareParameter("useThreeChords", "consider only the 3 main triad chords of the key (T, D, SD) to build the polyphonic profiles", "{true,false}", true);
    declareParameter("numHarmonics", "number of harmonics that should contribute to the polyphonic profile (1 only considers the fundamental harmonic)", "[1,inf)", 4);
    declareParameter("slope", "value of the slope of the exponential harmonic contribution to the polyphonic profile", "[0,inf)", 0.6);
    declareParameter("profileType", "the type of polyphic profile to use for correlation calculation", "{diatonic,krumhansl,temperley,weichai,tonictriad,temperley2005,thpcp,shaath,gomez,noland,edmm,edma,bgate,braw}", "bgate");
    declareParameter("pcpSize", "number of array elements used to represent a semitone times 12 (this parameter is only a hint, during computation, the size of the input PCP is used instead)", "[12,inf)", 36);
    declareParameter("useMajMin", "use a third profile called 'majmin' for ambiguous tracks [4]. Only avalable for the edma, bgate and braw profiles", "{true,false}", false);
  }

  void compute();
  void configure();

  static const char* name;
  static const char* category;
  static const char* description;

protected:
  enum Scales {
    MAJOR  = 0,
    MINOR  = 1,
    MAJMIN = 2
  };

  std::vector<Real> _m;
  std::vector<Real> _M;
  std::vector<Real> _O;

  std::vector<Real> _profile_doM;
  std::vector<Real> _profile_dom;
  std::vector<Real> _profile_doO;

  Real _mean_profile_M;
  Real _mean_profile_m;
  Real _mean_profile_O;

  Real _std_profile_M;
  Real _std_profile_m;
  Real _std_profile_O;

  Real _slope;
  int _numHarmonics;
  std::string _profileType;

  std::vector<std::string> _keys;
  bool _useMajMin;

  Real correlation(const std::vector<Real>& v1, const Real mean1, const Real std1, const std::vector<Real>& v2, const Real mean2, const Real std2, const int shift) const;
  void addContributionHarmonics(const int pitchclass, const Real contribution, std::vector<Real>& M_chords) const;
  void addMajorTriad(const int root, const Real contribution, std::vector<Real>& M_chords) const;
  void addMinorTriad(int root, Real contribution, std::vector<Real>& M_chords) const;
  void resize(int size);
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmcomposite.h"
#include "pool.h"

namespace essentia {
namespace streaming {

class Key : public AlgorithmComposite {
 protected:
  Sink<std::vector<Real> > _pcp;

  Source<std::string> _key;
  Source<std::string> _scale;
  Source<Real> _strength;

  Pool _pool;
  Algorithm* _poolStorage;
  standard::Algorithm* _keyAlgo;

  bool _averageDetuningCorrection;
  Real _pcpThreshold;

  void normalizePcpPeak(std::vector<Real>& pcp);
  void pcpGate(std::vector<Real>& pcp, Real threshold);
  void shiftPcp(std::vector<Real>& pcp);


 public:
  Key();
  ~Key();

  void declareParameters() {
    declareParameter("usePolyphony", "enables the use of polyphonic profiles to define key profiles (this includes the contributions from triads as well as pitch harmonics)", "{true,false}", true);
    declareParameter("useThreeChords", "consider only the 3 main triad chords of the key (T, D, SD) to build the polyphonic profiles", "{true,false}", true);
    declareParameter("numHarmonics", "number of harmonics that should contribute to the polyphonic profile (1 only considers the fundamental harmonic)", "[1,inf)", 4);
    declareParameter("slope", "value of the slope of the exponential harmonic contribution to the polyphonic profile", "[0,inf)", 0.6);
    declareParameter("profileType", "the type of polyphic profile to use for correlation calculation", "{diatonic,krumhansl,temperley,weichai,tonictriad,temperley2005,thpcp,shaath,gomez,noland,edmm,edma,bgate,braw}", "bgate");
    declareParameter("pcpSize", "number of array elements used to represent a semitone times 12 (this parameter is only a hint, during computation, the size of the input PCP is used instead)", "[12,inf)", 36);
    declareParameter("pcpThreshold", "pcp bins below this value are set to 0", "[0,1]", 0.2);
    declareParameter("averageDetuningCorrection", "shifts a pcp to the nearest tempered bin", "{true,false}", true);
    declareParameter("useMajMin", "songs can also be clasified as MajMin [4]. Only avalable for the edma, bgate and braw profiles", "{true,false}", false);
  }

  void configure();

  void declareProcessOrder() {
    declareProcessStep(SingleShot(_poolStorage));
    declareProcessStep(SingleShot(this));
  }

  AlgorithmStatus process();
  void reset();

  static const char* name;
  static const char* category;
  static const char* description;

};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_KEY_H
