/*
 * Copyright (C) 2006-2012 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "pitchsaliencefunction.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PitchSalienceFunction::name = "PitchSalienceFunction";
const char* PitchSalienceFunction::version = "1.0";
const char* PitchSalienceFunction::description = DOC("This algorithm computes the pitch salience function of a signal frame given its spectral peaks. The salience function covers a pitch range of nearly five octaves (i.e., 6000 cents), starting from the \"referenceFrequency\", and is quantized into cent bins according to the specified \"binResolution\". The salience of a given frequency is computed as the sum of the weighted energies found at integer multiples (harmonics) of that frequency. \n"
"\n"
"This algorithm is intended to receive its \"frequencies\" and \"magnitudes\" inputs from the SpectralPeaks algorithm. The output is a vector of salience values computed for the cent bins.\n"
"\n"
"When input vectors differ in size or are empty, an exception is thrown. Input vectors must contain positive frequencies and not contain negative magnitudes otherwise an exception is thrown. It is highly recommended to avoid erroneous peak duplicates (peaks of the same frequency occuring more than ones), but it is up to the user's own control and no exception will be thrown.\n"
"\n"
"References:\n"
"  [1] J. Salamon and E. Gómez, \"Melody extraction from polyphonic music\n"
"  signals using pitch contour characteristics,\" IEEE Transactions on Audio,\n"
"  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.\n");

void PitchSalienceFunction::configure() {
  _referenceFrequency = parameter("referenceFrequency").toReal();
  _binResolution = parameter("binResolution").toReal();
  _magnitudeThreshold = parameter("magnitudeThreshold").toReal();
  _magnitudeCompression = parameter("magnitudeCompression").toReal();
  _numberHarmonics = parameter("numberHarmonics").toInt();
  _harmonicWeight = parameter("harmonicWeight").toReal();

  _numberBins = floor(6000.0 / _binResolution); // range of 5 octaves in cent bins
  _binsInSemitone = floor(100.0 / _binResolution);
  _binsInOctave = 1200.0 / _binResolution;
  _referenceTerm = 0.5 - _binsInOctave * log2(_referenceFrequency);
  _magnitudeThresholdLinear = 1.0 / pow(10.0, _magnitudeThreshold/20.0);

  _harmonicWeights.clear();
  _harmonicWeights.reserve(_numberHarmonics);
  for (int h=0; h<_numberHarmonics; h++) {
    _harmonicWeights.push_back(pow(_harmonicWeight, h));
  }

  _nearestBinsWeights.resize(_binsInSemitone + 1);
  for (int b=0; b <= _binsInSemitone; b++) {
    _nearestBinsWeights[b] = pow(cos((Real(b)/_binsInSemitone)* M_PI/2), 2);
  }
}

void PitchSalienceFunction::compute() {
  const vector<Real>& frequencies = _frequencies.get();
  const vector<Real>& magnitudes = _magnitudes.get();
  vector <Real>& salienceFunction = _salienceFunction.get();

  // do sanity checks
  if (magnitudes.size() != frequencies.size()) {
    throw EssentiaException("PitchSalienceFunction: frequency and magnitude input vectors must have the same size");
  }

  if (frequencies.empty()) {
    // no peaks -> return zero salience function
    salienceFunction.clear();
    salienceFunction.resize(_numberBins, 0.0);
    return;
  }

  int numberPeaks = frequencies.size();
  for (int i=0; i<numberPeaks; i++) {
    if (frequencies[i] <= 0) {
      throw EssentiaException("PitchSalienceFunction: spectral peak frequencies must be positive");
    }
    if (magnitudes[i] <= 0) {
      throw EssentiaException("PitchSalienceFunction: spectral peak magnitudes must be positive");
    }
  }


  salienceFunction.resize(_numberBins);
  fill(salienceFunction.begin(), salienceFunction.end(), (Real) 0.0);
  Real minMagnitude = magnitudes[argmax(magnitudes)] * _magnitudeThresholdLinear;

  for (int i=0; i<numberPeaks; i++) {
    // remove peaks with low magnitudes:
    // 20 * log10(magnitudes[argmax(magnitudes)]/magnitudes[i]) >= _magnitudeThreshold
    if (magnitudes[i] <= minMagnitude) {
      continue;
    }
    Real magnitudeFactor = pow(magnitudes[i], _magnitudeCompression);

    // find all bins where this peak contributes salience
    // these bins are (sub)harmonics of the peak frequency
    // propagate salience to nearest bins within +- one semitone

    for (int h=0; h<_numberHarmonics; h++) {
      int h_bin = frequencyToCentBin(frequencies[i] / (h+1));
      if (h_bin < 0) {
        break;
      }

      for(int b=max(0, h_bin-_binsInSemitone); b <= min(_numberBins-1, h_bin+_binsInSemitone); b++) {
        salienceFunction[b] += magnitudeFactor * _nearestBinsWeights[abs(b-h_bin)] * _harmonicWeights[h];
      }
    }

  }
}

int PitchSalienceFunction::frequencyToCentBin(Real frequency) {
  // +0.5 term is used instead of +1 (as in [1]) to center 0th bin to 55Hz
  // formula: floor(1200 * log2(frequency / _referenceFrequency) / _binResolution + 0.5)
  //    --> 1200 * (log2(frequency) - log2(_referenceFrequency)) / _binResolution + 0.5
  //    --> 1200 * log2(frequency) / _binResolution + (0.5 - 1200 * log2(_referenceFrequency) / _binResolution)
  return floor(_binsInOctave * log2(frequency) + _referenceTerm);
}

