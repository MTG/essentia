/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "temposcalebands.h"
#include "essentiamath.h" // log10

using namespace std;
using namespace essentia;
using namespace standard;


const char* TempoScaleBands::name = "TempoScaleBands";
const char* TempoScaleBands::description = DOC("This algorithm computes features for tempo tracking. The output features should be used with the tempotap algorithm. See standard_tempotapExtractor in examples folder.\n"
"\n"
"An exception is thrown if less than 1 band is given. An exception is also thrown if the there are not an equal number of bands given as band-gains given.\n"
"\n"
"References:\n"
"  [1] Innovative algorithm by Fabien Gouyon and Simon Dixon. There is no reference at the time of this writing.\n");

void TempoScaleBands::compute() {
  const vector<Real>& bands = _bands.get();
  vector<Real>& scaledBands = _scaledBands.get();
  Real& cumul = _cumulBands.get();

  int size = bands.size();
  if (size < 1) {
    throw EssentiaException("TempoScaleBands: a power spectrum should have 1 band, at least");
  }

  if ((int)_bandsGain.size() != size) {
    throw EssentiaException("TempoScaleBands: bandsGain and bands have different sizes");
  }

  scaledBands.resize(size);
  _scratchBands.resize(size);
  _oldBands.resize(size);

  for (int i=0; i<size; i++) {
    scaledBands[i] = log10(1.0 + 100.0 * bands[i]) / log10(101.0);
  }

  cumul = 0.0;
  for (int i=0; i<size; i++) {
    _scratchBands[i] = max((Real)0.0, scaledBands[i]-_oldBands[i]) * _frameFactor;
    cumul += _scratchBands[i];
  }
  cumul = scale(cumul, 0.2, 1.2, 0.3);

  for (int i=0; i<size; i++) {
    _oldBands[i] = scaledBands[i];
    scaledBands[i] = scale(_scratchBands[i], 0.1, 0.5, 0.4);
    scaledBands[i] *= _bandsGain[i];
  }
}

void TempoScaleBands::configure() {
  _frameFactor = sqrt( 256. / parameter("frameTime").toReal() );
  _bandsGain = parameter("bandsGain").toVectorReal();
  if (_bandsGain.size() == 0) {
    throw EssentiaException("TempoScaleBands: bandsGain should have 1 gain, at least");
  }
  reset();
}

Real TempoScaleBands::scale(const Real& value, const Real& c1, const Real& c2, const Real& pwr) {
  if (value > c2) {
    return c2 + 0.1 * log10(value / c2);
  }
  if (value > c1) {
    return c2 + (c2 - c1) * pow((value - c1) / (c2 - c1), pwr);
  }
  return value;
}

void TempoScaleBands::reset() {
  for (int i=0; i<int(_oldBands.size()); ++i) {
    _oldBands[i] = 0.0;
  }
}
