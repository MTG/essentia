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

#include "windowing.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Windowing::name = "Windowing";
const char* Windowing::description = DOC("This algorithm applies windowing to audio signals.\n"
"It optionally applies zero-phase windowing and optionally adds zero-padding.\n"
"The resulting windowed frame size is equal to the incoming frame size plus the number of padded zeros.\n"
"The available windows are normalized (to have an area of 1) and then scaled by a factor of 2.\n"
"\n"
"An exception is thrown if the size of the frame is less than 2.\n"
"\n"
"References:\n"
"  [1] F. J. Harris, \"On the use of windows for harmonic analysis with the\n"
"  discrete Fourier transform, Proceedings of the IEEE, vol. 66, no. 1,\n"
"  pp. 51-83, Jan. 1978\n\n"
"  [2] Window function - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Window_function");

void Windowing::configure() {
  _window.resize(parameter("size").toInt());
  createWindow(parameter("type").toLower());

  _zeroPadding = parameter("zeroPadding").toInt();
  _zeroPhase = parameter("zeroPhase").toBool();
}

void Windowing::createWindow(const std::string& windowtype) {
  if (windowtype == "hamming") hamming();
  else if (windowtype == "hann") hann();
  else if (windowtype == "triangular") triangular();
  else if (windowtype == "square") square();
  else if (windowtype == "blackmanharris62") blackmanHarris62();
  else if (windowtype == "blackmanharris70") blackmanHarris70();
  else if (windowtype == "blackmanharris74") blackmanHarris74();
  else if (windowtype == "blackmanharris92") blackmanHarris92();

  normalize();
}

void Windowing::compute() {
  const std::vector<Real>& signal = _frame.get();
  std::vector<Real>& windowedSignal = _windowedFrame.get();

  if (signal.size() <= 1) {
    throw EssentiaException("Windowing: frame size should be larger than 1");
  }

  if (signal.size() != _window.size()) {
    _window.resize(signal.size());
    createWindow(parameter("type").toLower());
  }

  int signalSize = (int)signal.size();
  int totalSize = signalSize + _zeroPadding;

  windowedSignal.resize(totalSize);

  int i = 0;

  if (_zeroPhase) {
    // first half of the windowed signal is the
    // second half of the signal with windowing!
    for (int j=signalSize/2; j<signalSize; j++) {
      windowedSignal[i++] = signal[j] * _window[j];
    }

    // zero padding
    for (int j=0; j<_zeroPadding; j++) {
      windowedSignal[i++] = 0.0;
    }

    // second half of the signal
    for (int j=0; j<signalSize/2; j++) {
      windowedSignal[i++] = signal[j] * _window[j];
    }
  }
  else {
    // windowed signal
    for (int j=0; j<signalSize; j++) {
      windowedSignal[i++] = signal[j] * _window[j];
    }

    // zero padding
    for (int j=0; j<_zeroPadding; j++) {
      windowedSignal[i++] = 0.0;
    }
  }
}

// values which were 0.54 and 0.46 are actually approximations.
// More precise values are 0.53836 and 0.46164 (found on wikipedia)
// @todo find a more "scientific" reference than wikipedia
void Windowing::hamming() {
  const int size = _window.size();

  for (int i=0; i<size; i++) {
    _window[i] = 0.53836 - 0.46164 * cos((2.0*M_PI*i) / (size - 1.0));
  }
}

void Windowing::hann() {
  const int size = _window.size();

  for (int i=0; i<size; i++) {
    _window[i] = 0.5 - 0.5 * cos((2.0*M_PI*i) / (size - 1.0));
  }
}

// note: this window has non-zero end-points, if you want zero end-points, you will need a bartlett window
void Windowing::triangular() {
  int size = int(_window.size());

  for (int i=0; i<size; i++) {
    _window[i] = 2.0/size * (size/2.0 - abs((Real)(i - (size-1.)/2.)));
  }
}

void Windowing::square() {
  for (int i=0; i<int(_window.size()); i++) {
    _window[i] = 1.0;
  }
}


// @todo lookup implementation of windows on wikipedia and other resources
void Windowing::blackmanHarris(double a0, double a1, double a2, double a3) {
  int size = _window.size();

  double fConst = 2.0 * M_PI / (size-1);

  if (size % 2 !=0) {
    _window[size/2] = a0 - a1 * cos(fConst * (size/2)) + a2 *
      cos(fConst * 2 * (size/2)) - a3 * cos(fConst * 3 * (size/2));
  }

  for (int i=0; i<size/2; i++) {
    _window[i] = _window[size-i-1] = a0 - a1 * cos(fConst * i) +
      a2 * cos(fConst * 2 * i) - a3 * cos(fConst * 3 * i);
  }
}

void Windowing::blackmanHarris62() {
  double a0 = .44959, a1 = .49364, a2 = .05677;
  blackmanHarris(a0, a1, a2);
}

void Windowing::blackmanHarris70() {
  double a0 = .42323, a1 = .49755, a2 = .07922;
  blackmanHarris(a0, a1, a2);
}

void Windowing::blackmanHarris74() {
  double a0 = .40217, a1 = .49703, a2 = .09892, a3 = .00188;
  blackmanHarris(a0, a1, a2, a3);
}

void Windowing::blackmanHarris92() {
  double a0 = .35875, a1 = .48829, a2 = .14128, a3 = .01168;
  blackmanHarris(a0, a1, a2, a3);
}


void Windowing::normalize() {
  const int size = _window.size();
  Real sum = 0.0;
  for (int i=0; i<size; i++) {
    sum += abs(_window[i]);
  }

  if (sum == 0.0) {
    return;
  }

  // as we have half of the energy in negative frequencies, we need to scale, but
  // multiply by two. Otherwise a sinusoid at 0db will result in 0.5 in the spectrum.
  Real scale = 2.0 / sum;

  for (int i=0; i<size; i++) {
    _window[i] *= scale;
  }
}
