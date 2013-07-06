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

#include "aftermaxtobeforemaxenergyratio.h"
#include "essentiamath.h"
using namespace std;

namespace essentia {
namespace standard {

const char* AfterMaxToBeforeMaxEnergyRatio::name = "AfterMaxToBeforeMaxEnergyRatio";
const char* AfterMaxToBeforeMaxEnergyRatio::description = DOC("This algorithm calculates the ratio between the pitch energy after the pitch maximum and the pitch energy before the pitch maximum. Sounds having an monotonically ascending pitch or one unique pitch will show a value of (0,1], while sounds having a monotonically descending pitch will show a value of [1,inf). In case there is no energy before the max pitch, the algorithm will return the energy after the maximum pitch.\n"
"\n"
"The algorithm throws exception when input is either empty or contains only zeros.");

void AfterMaxToBeforeMaxEnergyRatio::compute() {

  vector<Real> pitch = _pitch.get();
  Real& afterMaxToBeforeMaxEnergyRatio = _afterMaxToBeforeMaxEnergyRatio.get();

  // Remove all 0Hz elements
  vector<Real>::iterator i = pitch.begin();
  while (i != pitch.end()) {
    if (*i <= 0.0) {
      i = pitch.erase(i);
    }
    else {
      i++;
    }
  }

  if (pitch.empty()) {
    throw EssentiaException("AfterMaxToBeforeMaxEnergyRatio: pitch array doesn't contain any non-zero values or is empty");
  }

  int nMax = argmax(pitch);
  Real energyBeforeMax = 0.0;
  Real energyAfterMax = 0.0;
  for (int i=0; i<=nMax; ++i) {
    energyBeforeMax += pitch[i] * pitch[i];
  }
  for (int i=nMax; i<int(pitch.size()); ++i) {
    energyAfterMax += pitch[i] * pitch[i];
  }

  // division by zero will never occure as first we have removed any elements with 0Hz
  // and the max pitch is included in both the energy before and energy after.
  afterMaxToBeforeMaxEnergyRatio = energyAfterMax / energyBeforeMax;
}

} // namespace standard
} // namespace essentia


namespace essentia {
namespace streaming {

const char* AfterMaxToBeforeMaxEnergyRatio::name = essentia::standard::AfterMaxToBeforeMaxEnergyRatio::name;
const char* AfterMaxToBeforeMaxEnergyRatio::description = essentia::standard::AfterMaxToBeforeMaxEnergyRatio::description;

AlgorithmStatus AfterMaxToBeforeMaxEnergyRatio::process() {
  // TODO: can be optimized with a lookup on _pitch.available() (similar to poolstorage & vectorinput)
  while (_pitch.acquire(1)) {
    _accu.push_back(_pitch.firstToken());
    _pitch.release(1);
  }

  if (!shouldStop()) return NO_INPUT;

  // this should go into the constructor
  standard::Algorithm* afterMaxToBeforeMaxEnergyRatio = standard::AlgorithmFactory::create("AfterMaxToBeforeMaxEnergyRatio");
  Real ratio = 0;
  afterMaxToBeforeMaxEnergyRatio->input("pitch").set(_accu);
  afterMaxToBeforeMaxEnergyRatio->output("afterMaxToBeforeMaxEnergyRatio").set(ratio);
  afterMaxToBeforeMaxEnergyRatio->compute();
  delete afterMaxToBeforeMaxEnergyRatio;

  _afterMaxToBeforeMaxEnergyRatio.push(ratio);

  return FINISHED;
}

} // namespace streaming
} // namespace essentia
