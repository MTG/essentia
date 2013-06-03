/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "clipper.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Clipper::name = "Clipper";
const char* Clipper::description = DOC("This algorithm clips the input signal to fit between the range given by the min and max parameters.\n"
"\n"
"References:\n"
"  [1] Clipping - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Clipping_(audio)");

void Clipper::configure() {
  _max = parameter("max").toReal();
  _min = parameter("min").toReal();
}

void Clipper::compute() {
  const std::vector<Real>& input = _input.get();
  std::vector<Real>& output = _output.get();
  int size = input.size();
  output.resize(size);
  for (int i = 0; i < size; ++i) {
    if (input[i] > _max) output[i] = _max;
    else if (input[i] < _min) output[i] = _min;
    else output[i] = input[i];
  }
}
