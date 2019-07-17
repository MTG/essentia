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

#include "welch.h"

using namespace std;

#include "poolstorage.h"

namespace essentia {
namespace standard {

const char* Welch::name = "Welch";
const char* Welch::category = "Standard";
const char* Welch::description = DOC(" This algorithm estimates the Power Spectral Density of the input signal using"
" the Welch's method [1].\n The input should be fed with the overlapped audio frames. The algorithm stores internally the"
"required past frames to compute each output. Call reset() to clear the buffers. This implentation is based on Scipy [2]\n"
"\n"
"References:\n"
"  [1] The Welch's method - Wikipedia, the free encyclopedia,\n"
       "https://en.wikipedia.org/wiki/Welch%27s_method\n"
"  [2] https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.welch.html");

void Welch::configure() {
  _sampleRate = parameter("sampleRate").toReal();
  _windowType = parameter("windowType").toString();
  _fftSize = nextPowerTwo(parameter("fftSize").toInt());
  _scaling = parameter("scaling").toString();
  _averagingFrames = parameter("averagingFrames").toInt();
  _frameSize = parameter("frameSize").toInt();

  initBuffers();
};

void Welch::initBuffers() {
  if (_frameSize > _fftSize) {
    _fftSize = nextPowerTwo(_frameSize);
    E_INFO("Welch: fftSize has to be power of 2 and greater than frameSize. Resizing to " << _fftSize << " samples.");
  }

  _padding = _fftSize - _frameSize;

  _spectSize = _fftSize / 2 + 1;

  _window->configure("size", _frameSize,
                    "zeroPadding", _padding,
                    "type", _windowType,
                    "normalized", false);
  _window->output("frame").set(_windowed);

  _powerSpectrum->configure("size", _fftSize);
  
  _powerSpectrum->output("powerSpectrum").set(_powerSpectrumFrame);

  _psdBuffer.assign(_averagingFrames, vector<Real>(_spectSize, 0.f));

  vector<Real> ones( _fftSize, 1.f);
  _window->input("frame").set(ones);
  _window->compute();

  if (_scaling == "density")
    _normalization =  1.f / (_sampleRate * energy(_windowed) * (Real)_averagingFrames);
  
  if (_scaling == "power")
    _normalization = 1.f / (pow(sum(_windowed), 2.f) * (Real)_averagingFrames);
};

void Welch::compute() {
  const vector<Real>& frame = _frame.get();
  std::vector<Real>& psd = _psd.get();
  psd.assign(_spectSize, 0.f);

  if (frame.size() != _frameSize) {
    E_INFO("Welch: frameSize was configured to " << _frameSize << " but encountered " <<
           frame.size() << " samples on running time. Resizing buffers.");    
    initBuffers();
  }

  _window->input("frame").set(frame);
  _window->compute();

  _powerSpectrum->input("signal").set(_windowed);
  _powerSpectrum->compute();
  
  for (uint j = 0; j < _spectSize; j++) {
    _powerSpectrumFrame[j] *= _normalization;
    if ((j>0) && (j<_spectSize-1))
      _powerSpectrumFrame[j] *= 2.f;
    }

  rotate(_psdBuffer.begin(), _psdBuffer.begin() + 1, _psdBuffer.end());

  fastcopy(&_psdBuffer[_averagingFrames - 1][0], &_powerSpectrumFrame[0], _spectSize);

  for (uint j = 0; j < _spectSize; j++)
    for (uint i = 0; i < _averagingFrames; i++)
      psd[j] += _psdBuffer[i][j];
}

void Welch::reset() {
  initBuffers();
}

} // namespace standard
} // namespace essentia


