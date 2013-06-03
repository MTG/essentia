/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
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
