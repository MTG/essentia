/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "tuningfrequency.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* TuningFrequency::name = "TuningFrequency";
const char* TuningFrequency::description = DOC("Given a sequence/set of spectral peaks, this algorithm estimates the tuning frequency of a given song. The result is the tuning frequency in Hz, and its distance from 440Hz in cents. This version is slightly adapted from the original algorithm by Emilia Gomez, but gives the same results.\n"
"\n"
"Input vectors should have the same size, otherwise an exception is thrown. This algorithm should be given the outputs of the spectral peaks algorithm.\n"
"\n"
"References:\n"
"  [1] E. Gómez, Key Estimation from Polyphonic Audio,\n"
"      Music Technology Group, Pompeu Fabra University, 2005");

const Real TuningFrequency::wrappingBoundary = -35;

void TuningFrequency::configure() {
  _resolution = parameter("resolution").toReal();
  reset();
}

void TuningFrequency::reset() {
  int size = (int)(100.0/_resolution);
  _histogram = vector<Real>(size, 0.0);
  _globalHistogram = vector<Real>(size, 0.0);
}

Real TuningFrequency::currentTuningCents() const {
  int globalIndex = argmax(_globalHistogram);

  // if we have an empty histogram (ie: no estimates atm), start with a default
  // value of everything is tuned correctly
  if (_globalHistogram[globalIndex] == (Real)0.0) {
    return 0.0;
  }

  Real tuningCents = _resolution*globalIndex - 50.0;

  // post-processing to avoid getting confused between tc=-50 and tc=50
  // (which are fundamentally the same) by resetting the origin for the ref frequency.
  if (tuningCents < wrappingBoundary) {
    tuningCents += 100;
  }

  return tuningCents;
}

void TuningFrequency::updateOutputs() {
  Real& tuningCents = _tuningCents.get();
  Real& tuningFrequency = _tuningFrequency.get();

  tuningCents = currentTuningCents();
  tuningFrequency = tuningFrequencyFromCents(tuningCents);
}


Real TuningFrequency::tuningFrequencyFromCents(Real cents) const {
  return 440.0*pow((Real)2.0, (Real)(cents/1200.0));
}


void TuningFrequency::compute() {
  const vector<Real>& frequencies = _frequencies.get();
  const vector<Real>& magnitudes = _magnitudes.get();

  if (magnitudes.size() != frequencies.size()) {
    throw EssentiaException("TuningFrequency: Frequency and magnitude vector have different size");
  }

  if (magnitudes.empty()) {
    // if we have no peaks, just return the same value as before
    updateOutputs();
    return;
  }

  // the frame histogram is reset every frame...
  fill(_histogram.begin(), _histogram.end(), (Real) 0.0);

  // the peak energy
  Real frame_energy = (Real)0.0;

  // Compute histogram with arbitrary cents resolution
  for (int i=0; i<(int)magnitudes.size(); i++) {
    if (frequencies[i] <= 0.0) {
      continue;
    }

    Real octave = log2(frequencies[i]/440.0);
    Real note = octave*12.f;
    Real deviationInCents = 100.0*(note - floor(note + 0.5));
    int index = int((50. + deviationInCents) / _resolution + 0.5);

    if (index == int(_histogram.size())) {
      index = 0;
    }
    //if (index >= (int)_histogram.size() || index < 0) { // this case will never occur
    //  throw EssentiaException("TuningFrequency: Index smaller or equal to zero.");
    //}


    _histogram[index] += magnitudes[i];
  }

  frame_energy = energy(magnitudes);

  // Compute 'frame' maximum histogram value as the tuning of the frame
  vector<Real>::iterator frameMaxElement = max_element(_histogram.begin(), _histogram.end());
  int frameIndex = frameMaxElement - _histogram.begin();
  Real frameTuning = _resolution*frameIndex - 50.0;

  // Compute 'global' maximum histogram value, i.e. the sum of all!
  // this is a bit strange, as we only want the "last" value that is
  // computed.
  int globalHistogramIndex = (int) ((50.0+frameTuning)/_resolution+0.5);

  if (globalHistogramIndex == (int)_globalHistogram.size()) {
    globalHistogramIndex = 0;
  }
  _globalHistogram[globalHistogramIndex] += frame_energy;

  updateOutputs();
}
