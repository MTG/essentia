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

#ifndef PEAK_H
#define PEAK_H

#include <utility> // std::pair
#include "types.h"

namespace essentia {
namespace util {

class Peak {
  public:
    Real position;
    Real magnitude;

  Peak() : position(), magnitude() {}
  Peak(const Peak& p) : position(p.position), magnitude(p.magnitude) {}

  template<typename T, typename U>
  Peak(const T& pos, const U& mag) : position(pos), magnitude(mag) {}

  template<typename T, typename U>
  Peak(const std::pair<T,U>& p) : position(p.first), magnitude(p.second) {}

  bool operator ==(const Peak& p) const {
    return (position == p.position) && (magnitude == p.magnitude);
  }

  bool operator !=(const Peak& p) const {
    return (position != p.position) || (magnitude != p.magnitude);
  }

  bool operator< (const Peak& p) const { return magnitude <  p.magnitude; }
  bool operator> (const Peak& p) const { return magnitude >  p.magnitude; }
  bool operator<=(const Peak& p) const { return magnitude <= p.magnitude; }
  bool operator>=(const Peak& p) const { return magnitude >= p.magnitude; }

  Peak& operator=(const Peak& p) {
    position = p.position; magnitude = p.magnitude;
    return *this;
  }

  template<typename T, typename U>
  Peak& operator=(const std::pair<T, U>& p) {
    position = p.first;magnitude = p.second;
    return *this;
  }
};

// peak comparison:

// comparing by position, by default sorts by ascending position and in case
// the positions are equal it sorts by descending magnitude
template<typename Comp1=std::less<Real>,
         typename Comp2=std::greater_equal<Real> >
class ComparePeakPosition : public std::binary_function<Real, Real, bool> {
  Comp1 _cmp1;
  Comp2 _cmp2;
  public:
    bool operator () (const Peak& p1, const Peak& p2) const {
      if (_cmp1(p1.position, p2.position)) return true;
      if (_cmp1(p2.position, p1.position)) return false;
      return _cmp2(p1.magnitude, p2.magnitude);
    }
};

// comparing by magnitude, by default sorts by descending magnitude and in case
// the magnitudes are equal it sorts by ascending position
template<typename Comp1=std::greater<Real>,
         typename Comp2=std::less_equal<Real> >
class ComparePeakMagnitude : public std::binary_function<Real, Real, bool> {
  Comp1 _cmp1;
  Comp2 _cmp2;
  public:
    bool operator () (const Peak& p1, const Peak& p2) const {
      if (_cmp1(p1.magnitude, p2.magnitude)) return true;
      if (_cmp1(p2.magnitude, p1.magnitude)) return false;
      return _cmp2(p1.position, p2.position);
    }
};

// from 2 vector<Real> to vector<Peak>:
inline std::vector<Peak> realsToPeaks(const std::vector<Real>& pos,
                                      const std::vector<Real>& mag) {
  int size = pos.size();
  if (size != int(mag.size())) {
      throw EssentiaException("realsToPeaks: position vector size != magnitude vector size");
  }
  std::vector<Peak> peaks(size);
  for (int i=0; i<size; i++) {
    peaks[i] = Peak(pos[i], mag[i]);
  }
  return peaks;
}

// from vector<Peak> to 2 vector<Real>
inline void peaksToReals(const std::vector<Peak>& peaks,
                         std::vector<Real>& pos, std::vector<Real>& mag) {
  int size = peaks.size();
  if (size != int(pos.size())) pos.resize(size);
  if (size != int(mag.size())) mag.resize(size);

  for (int i=0; i<size; i++) {
    pos[i] = peaks[i].position;
    mag[i] = peaks[i].magnitude;
  }
}

inline std::ostream& operator<<(std::ostream& out, const Peak& p) {
  return out << "position: " << p.position << ", magnitude: " << p.magnitude;
}

} // namespace util
} // namespace essentia

#endif // PEAK_H
