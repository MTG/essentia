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

#ifndef BETOOLS_H
#define BETOOLS_H

#include <algorithm>
#include <fstream>
#include <cassert>
#include "types.h"

// This file contains serialization helper functions that should be aware of
// endianness: they assume all streams are big-endian (network-order),
// regardless of the architecture

namespace essentia {

inline bool isBigEndian() {
  const int i = 1;
  return (*(char*)&i) == 0;
}

template <typename T>
void removeEndianness(T& x) {
  if (isBigEndian()) return;
  char* p = (char*)&x;
  std::reverse(p, p+sizeof(T));
}



template <typename T>
void bewrite(std::ofstream& out, const T& value) {
  T becopy(value);
  removeEndianness(becopy);
  out.write((char*)&becopy, sizeof(becopy));
}

template <>
void bewrite(std::ofstream& out, const std::string& str) {
  bewrite(out, (sint32)str.size());
  out.write(str.c_str(), (int)str.size());
}

template <>
void bewrite(std::ofstream& out, const std::vector<std::string>& v) {
  sint32 size = (sint32)v.size();
  bewrite(out, size);
  for (int i=0; i<size; i++) bewrite(out, v[i]);
}



template <typename T>
void beread(std::ifstream& in, T& value) {
  in.read((char*)&value, sizeof(value));
  removeEndianness(value);
}

template <>
void beread(std::ifstream& in, std::string& str) {
  sint32 size;
  beread(in, size);
  str.resize(size);
  in.read(&str[0], size);
}

template <>
void beread(std::ifstream& in, std::vector<std::string>& v) {
  sint32 size;
  beread(in, size);
  v.resize(size);
  for (int i=0; i<size; i++) beread(in, v[i]);
}

} // namespace essentia

#endif // BETOOLS_H
