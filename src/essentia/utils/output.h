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

#ifndef ESSENTIA_OUTPUT_H
#define ESSENTIA_OUTPUT_H

#include <fstream>

namespace essentia {

  void outputYAMLArray(std::ostream& out, const std::vector<Real>& v) {
    out.precision(10);

    if (v.empty()) {
      out << "[]\n";
      return;
    }

    if (v.size() == 1) {
      out << v[0] << '\n';
      return;
    }

    // print 4 values/line
    out << "[ ";
    out.width(12);
    out << v[0];
    for (int i=1; i<(int)v.size(); i++) {
      // to newline or not to newline, that is the question...
      if (i%4 == 0) {
        out << ",\n                 ";
      }
      else {
        out << ",  ";
      }
      out.width(12);
      out << v[i];
    }
    out << "]";
  }

  void outputYAMLMatrix(std::ostream& out, const std::vector<std::vector<Real> >& v) {
    out.precision(10);

    out << "[ [ ";
    out.width(12);
    out << v[0][0];
    for (int j=1; j<(int)v[0].size(); j++) {
      out << ",  ";
      out.width(12);
      out << v[0][j];
    }
    out << "]";

    for (int i=1; i<(int)v.size(); i++) {
      out << ",\n            [ ";
      out.width(12);
      out << v[i][0];
      for (int j=1; j<(int)v[i].size(); j++) {
        out << ",  ";
        out.width(12);
        out << v[i][j];
      }
      out << "]";
    }
    out << " ]\n";

  }

} // namespace essentia

#endif // ESSENTIA_OUTPUT_H
