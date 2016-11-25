/*
 * spectrumtocent.cpp
 *
 *  Created on: Oct 26, 2016
 *      Author: pablo
 */

#include "spectrumtocent.h"
#include "essentiamath.h"

namespace essentia {
namespace standard {

const char* SpectrumToCent::name = "SpectrumToCent";
const char* SpectrumToCent::category = "Spectral";
const char* SpectrumToCent::description = DOC("This algorithm computes energy in triangular frequency bands of a spectrum equally spaced on the cent scale. Each band is computed to have a constant wideness in the cent scale. For each band the power-spectrum (mag-squared) is summed.\n"
"\n"
"Parameter \"centBinResolution\" should be and integer greater than 1, otherwise an exception will be thrown. TriangularBands is only defined for spectrum, which size is greater than 1.\n");


void SpectrumToCent::configure() {

  _sampleRate = parameter("sampleRate").toReal();
  _minFrequency = parameter("minimumFrequency").toReal();

  if ( _minFrequency >= _sampleRate / 2 ) {
    throw EssentiaException("SpectrumToCent: 'minimumFrequency' parameter is out of the range (0 - fs/2)");
  }

  _centBinRes= parameter("centBinResolution").toReal();
  _nBands= parameter("bands").toReal();

  calculateFilterFrequencies();

  if ( _bandFrequencies.back() > _sampleRate / 2 ) {
    E_INFO("Attempted to create bands up to " << _bandFrequencies.back() << "Hz with a Nyquist frequency of " << _sampleRate / 2 << "Hz.");
    throw EssentiaException("SpectrumToCent: Band frequencies cannot be above the Nyquist frequency.");
  }

  _triangularBands->configure(INHERIT("inputSize"),
                              INHERIT("sampleRate"),
                              "frequencyBands", _bandFrequencies,
                              INHERIT("log"),
                              INHERIT("normalize"),
                              INHERIT("type"));
}


void SpectrumToCent::compute() {
  const vector<Real>& spectrum = _spectrumInput.get();
  vector<Real>& bands = _bandsOutput.get();
  vector<Real>& freqs = _freqOutput.get();

  if (spectrum.size() <= 1) {
    throw EssentiaException("SpectrumToCent: the size of the input spectrum is not greater than one");
  }

  Real frequencyScale = (_sampleRate / 2.0) / (spectrum.size() - 1);

  for (int i=0; i<_nBands; i++) {

    int startBin = int(_bandFrequencies[i] / frequencyScale + 0.5);
    int midBin = int(_bandFrequencies[i + 1] / frequencyScale + 0.5);
    int endBin = int(_bandFrequencies[i + 2] / frequencyScale + 0.5);

    // finished
    if (startBin >= int(spectrum.size())) break;

    // going to far
    if (endBin > int(spectrum.size())) endBin = spectrum.size();

    if ((midBin == startBin) || (midBin == endBin) || (endBin == startBin)) {
      throw EssentiaException("SpectrumToCent: the number of spectrum bins is insufficient to compute the band (",
                              _bandFrequencies[i+1], "Hz). Use zero padding to increase the number of FFT bins.");
    }
  }

  freqs.resize(_nBands);

  for (int i = 0; i<_nBands; ++i) {
    freqs[i]= _bandFrequencies[i+1];
  }

  _triangularBands->input("spectrum").set(spectrum);
  _triangularBands->output("bands").set(bands);
  _triangularBands->compute();

}


void SpectrumToCent::calculateFilterFrequencies() {

  _bandFrequencies.resize( _nBands + 2 );

  for (int i=-1; i<=_nBands ; ++i) {
    _bandFrequencies[i+1] = _minFrequency * pow( 2, _centBinRes * i / ( 1200.0 ) );
  }
}

} // namespace standard
} // namespace essentia
