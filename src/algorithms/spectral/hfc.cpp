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
 * You should have received a copy of the GNU General Public License along with 
 * this program.  If not, see http://www.gnu.org/licenses/
 */

#include "hfc.h"

using namespace essentia;
using namespace standard;

const char* HFC::name = "HFC";
const char* HFC::description = DOC("This algorithm computes the High Frequency Content of a signal spectrum.\n"
"It can be computed according to the following techniques:\n"
"  - 'Masri' (default) which does: sum |X(n)|^2*k,\n"
"  - 'Jensen' which does: sum |X(n)|*k^2\n"
"  - 'Brossier' which does: sum |X(n)|*k\n"
"\n"
"Exception is thrown for empty input spectra.\n"
"\n"
"References:\n"
"  [1] P. Masri, A. Bateman, Improved Modelling of Attack Transients in Music Analysis-Resynthesis\n"
"      Digital Music Research Group, University of Bristol, 1996\n"
"  [2] K. Jensen, T. H. Anderson, Beat Estimation On The Beat\n"
"      Department of Computer Science, University of Copenhagen, 2003\n"
);

void HFC::configure() {
  _type = parameter("type").toLower();
  _sampleRate = parameter("sampleRate").toReal();
}

void HFC::compute() {

  // get the inputs and outputs
  const std::vector<Real>& spectrum = _spectrum.get();
  Real& hfc = _hfc.get();

  if (spectrum.size() == 0) {
    throw EssentiaException( "HFC: input audio spectrum empty" );
  }

  // Coefficient to convert bins into frequency
  Real bin2hz = 0.;

  if (spectrum.size() > 1) {
     bin2hz = (_sampleRate/2.0) / (Real)(spectrum.size() - 1);
  }

  // do the computation
  hfc = 0.0;

  // case "Masri" (default)
  if (_type == "masri") {
    for (std::vector<Real>::size_type i=0; i<spectrum.size(); i++) {
      hfc += (Real)i*bin2hz * spectrum[i] * spectrum[i];
    }
  }

  // case "Jensen"
  else if (_type == "jensen") {
    for (std::vector<Real>::size_type i=0; i<spectrum.size(); i++) {
      hfc += (Real)i*bin2hz * (Real)i*bin2hz * spectrum[i];
    }
  }

  // case "Brossier"
  else if (_type == "brossier") {
    for (std::vector<Real>::size_type i=0; i<spectrum.size(); i++) {
      hfc += (Real)i * bin2hz * spectrum[i];
    }
  }
}
