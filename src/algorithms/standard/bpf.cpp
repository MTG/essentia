/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "bpf.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* BPF::name = "BPF";
const char* BPF::description = DOC("A break point function linearly interpolates between discrete xy-coordinates to construct a continuous function.\n"
"\n"
"Exceptions are thrown when the size the vectors specified in parameters is not equal and at least they contain two elements. Also if the parameter vector for x-coordinates is not sorted ascendantly. A break point function cannot interpolate outside the range specified in parameter \"xPoints\". In that case an exception is thrown.\n "
"\n"
"References:\n"
"  [1] Linear interpolation - Wikipedia, the free encyclopedia,\n"
"      http://en.wikipedia.org/wiki/Linear_interpolation");


void BPF::compute() {
  const Real& xInput = _xInput.get();
  Real& yOutput = _yOutput.get();

  yOutput = bpf(xInput);
}


void BPF::configure() {
  bpf.init( parameter("xPoints").toVectorReal(), parameter("yPoints").toVectorReal());
}
