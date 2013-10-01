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

#include "crosscorrelation.h"

using namespace essentia;
using namespace standard;

const char* CrossCorrelation::name = "CrossCorrelation";
const char* CrossCorrelation::description = DOC("This algorithm computes the cross-correlation vector of two signals. It accepts 2 parameters, minLag and maxLag which define the range of the computation of the innerproduct.\n"
"\n"
"An exception is thrown if \"minLag\" is larger than \"maxLag\". An exception is also thrown if the input vectors are empty.\n"
"\n"
"References:\n"
"  [1] Cross-correlation - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Cross-correlation");

void CrossCorrelation::configure() {
  if (parameter("minLag").toInt() > parameter("maxLag").toInt()) {
    throw EssentiaException("CrossCorrelation: minLag parameter cannot be larger than maxLag parameter");
  }
}

void CrossCorrelation::compute() {

  const std::vector<Real>& signal_x = _signal_x.get();
  const std::vector<Real>& signal_y = _signal_y.get();
  std::vector<Real>& correlation = _correlation.get();

  if (signal_x.empty() || signal_y.empty()) {
    throw EssentiaException("CrossCorrelation: one or both of the input vectors are empty");
  }

  int wantedMinLag = parameter("minLag").toInt();
  int wantedMaxLag = parameter("maxLag").toInt();
  int minLag = std::max(wantedMinLag, -((int)signal_y.size() - 1));
  int maxLag = std::min(wantedMaxLag, (int)signal_x.size() - 1);

  int size = wantedMaxLag - wantedMinLag + 1;

  correlation.resize(size);

  int correlationIndex = 0;

  for (int i=0; i< minLag - wantedMinLag; i++) {
    correlation[correlationIndex++] = 0;
  }

  for (int lag = minLag; lag <= maxLag; lag++) {
    int i_start = std::max(0,lag);
    int i_end = std::min((int)signal_x.size(),(int)signal_y.size() + lag);
    Real corr = 0;

    for (int i=i_start; i<i_end; i++) {
      corr += signal_x[i] * signal_y[i - lag];
    }

    correlation[correlationIndex++] = corr;
  }

  for (int i=0; i<wantedMaxLag - maxLag; i++) {
    correlation[correlationIndex++] = 0;
  }
}
