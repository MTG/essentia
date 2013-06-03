/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "scale.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Scale::name = "Scale";
const char* Scale::description = DOC("This algorithm scales the audio by the specified factor, using clipping if required.");

void Scale::configure() {
  _factor = parameter("factor").toReal();
  _clipping = parameter("clipping").toBool();
  _maxValue = parameter("maxAbsValue").toReal();
}

void Scale::compute() {
  const vector<Real>& signal = _signal.get();
  vector<Real>& scaled = _scaled.get();

  scaled.resize(signal.size());
  fastcopy(scaled.begin(), signal.begin(), scaled.size());

  // scales first
  for (int i=0; i<(int)scaled.size(); i++) {
    scaled[i] *= _factor;
  }

  // does clipping, if applies
  if (_clipping) {
    for (int i=0; i<(int)scaled.size(); i++) {
      if (scaled[i] > _maxValue) scaled[i] = _maxValue;
      if (scaled[i] < -_maxValue) scaled[i] = -_maxValue;
    }
  }
}
