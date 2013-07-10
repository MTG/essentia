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

#ifndef ESSENTIA_WINDOWING_H
#define ESSENTIA_WINDOWING_H

#include "algorithm.h"

namespace essentia {
namespace standard {

class Windowing : public Algorithm {

 protected:
  Output<std::vector<Real> > _windowedFrame;
  Input<std::vector<Real> > _frame;

 public:
  Windowing() {
    declareInput(_frame, "frame", "the input audio frame");
    declareOutput(_windowedFrame, "frame", "the windowed audio frame");
  }

  void declareParameters() {
    declareParameter("size", "the window size", "[2,inf)", 1024);
    declareParameter("zeroPadding", "the size of the zero-padding", "[0,inf)", 0);
    declareParameter("type", "the window type, which can be 'hamming', 'hann', 'triangular', 'square' or 'blackmanharrisXX'", "{hamming,hann,triangular,square,blackmanharris62,blackmanharris70,blackmanharris74,blackmanharris92}", "hann");
    declareParameter("zeroPhase", "a boolean value that enables zero-phase windowing", "{true,false}", true);
  }

  void configure();

  void compute();

  static const char* name;
  static const char* description;

protected:
  void createWindow(const std::string& windowtype);

  // window generators
  void hamming();
  void hann();
  void triangular();
  void square();
  void normalize();
  void blackmanHarris(double a0, double a1, double a2, double a3 = 0.0);
  void blackmanHarris62();
  void blackmanHarris70();
  void blackmanHarris74();
  void blackmanHarris92();

  void makeZeroPhase();

  std::vector<Real> _window;
  int _zeroPadding;
  bool _zeroPhase;
};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class Windowing : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<Real> > _frame;
  Source<std::vector<Real> > _windowedFrame;

 public:
  Windowing() {
    declareAlgorithm("Windowing");
    declareInput(_frame, TOKEN, "frame");
    declareOutput(_windowedFrame, TOKEN, "frame");
  }
};

} // namespace streaming
} // namespace essentia

#endif // ESSENTIA_WINDOWING_H
