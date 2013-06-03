/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "spectralcomplexity.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* SpectralComplexity::name = "SpectralComplexity";
const char* SpectralComplexity::description = DOC("This algorithm computes the spectral complexity of an spectrum of Reals. The spectral complexity is based on the number of peaks in the input spectrum.\n"
"\n"
"It is recommended that the input \"spectrum\" be computed by the Spectrum algorithm. The input \"spectrum\" is passed to the SpectralPeaks algorithm and thus inherits its input requirements and exceptions.\n"
"References:\n"
"  [1] C. Laurier, O. Meyers, J. Serrà, M. Blech, P. Herrera, and X. Serra, Indexing music by mood: design and integration of an automatic content-based annotator,\n"
"      Multimedia Tools and Applications, vol. 48, no. 1, pp. 161–184, 2009.\n");


void SpectralComplexity::configure() {

  int sampleRate = parameter("sampleRate").toInt();
  Real magnitudeThreshold = parameter("magnitudeThreshold").toReal();

  _spectralPeaks->configure("sampleRate", sampleRate, "maxPeaks", 100, "maxFrequency", 5000, "minFrequency", 100, "magnitudeThreshold", magnitudeThreshold, "orderBy", "magnitude");

}

void SpectralComplexity::compute() {

  const vector<Real>& spectrum = _spectrum.get();
  Real& spectralComplexity = _spectralComplexity.get();

  vector<Real> frequencies;
  vector<Real> magnitudes;

  _spectralPeaks->input("spectrum").set(spectrum);
  _spectralPeaks->output("frequencies").set(frequencies);
  _spectralPeaks->output("magnitudes").set(magnitudes);
  _spectralPeaks->compute();

  spectralComplexity = (Real)magnitudes.size();
}
