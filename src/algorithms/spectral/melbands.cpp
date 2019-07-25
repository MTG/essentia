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

#include "melbands.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* MelBands::name = "MelBands";
const char* MelBands::category = "Spectral";
const char* MelBands::description = DOC("This algorithm computes energy in mel bands of a spectrum. It applies a frequency-domain filterbank (MFCC FB-40, [1]), which consists of equal area triangular filters spaced according to the mel scale. The filterbank is normalized in such a way that the sum of coefficients for every filter equals one. It is recommended that the input \"spectrum\" be calculated by the Spectrum algorithm.\n"
"\n"
"It is required that parameter \"highMelFrequencyBound\" not be larger than the Nyquist frequency, but must be larger than the parameter, \"lowMelFrequencyBound\". Also, The input spectrum must contain at least two elements. If any of these requirements are violated, an exception is thrown.\n"
"\n"
"Note: an exception will be thrown in the case when the number of spectrum bins (FFT size) is insufficient to compute the specified number of mel bands: in such cases the start and end bin of a band can be the same bin or adjacent bins, which will result in zero energy when summing bins for that band. Use zero padding to increase the number of spectrum bins in these cases.\n"
"\n"
"References:\n"
"  [1] T. Ganchev, N. Fakotakis, and G. Kokkinakis, \"Comparative evaluation\n"
"  of various MFCC implementations on the speaker verification task,\" in\n"
"  International Conference on Speach and Computer (SPECOM’05), 2005,\n"
"  vol. 1, pp. 191–194.\n\n"
"  [2] Mel-frequency cepstrum - Wikipedia, the free encyclopedia,\n"
"  http://en.wikipedia.org/wiki/Mel_frequency_cepstral_coefficient\n\n"
"  [3] Young, S. J., Evermann, G., Gales, M. J. F., Hain, T., Kershaw, D.,\n"
"  Liu, X., … Woodland, P. C. (2009). The HTK Book (for HTK Version 3.4).\n"
"  Construction, (July 2000), 384, https://doi.org/http://htk.eng.cam.ac.uk\n\n"
"  [4] Slaney, M. Auditory Toolbox: A MATLAB Toolbox for Auditory Modeling Work.\n"
"  Technical Report, version 2, Interval Research Corporation, 1998.");


void MelBands::configure() {
  if (parameter("highFrequencyBound").toReal() > parameter("sampleRate").toReal()*0.5 ) {
    throw EssentiaException("MelBands: High frequency bound cannot be higher than Nyquist frequency");
  }
  if (parameter("highFrequencyBound").toReal() <= parameter("lowFrequencyBound").toReal()) {
    throw EssentiaException("MelBands: High frequency bound cannot be lower than the low frequency bound.");
  }
  _numBands = parameter("numberBands").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _normalization = parameter("normalize").toString();
  _type = parameter("type").toString();

  setWarpingFunctions(parameter("warpingFormula").toString(),
                      parameter("weighting").toString());

  calculateFilterFrequencies();

  _triangularBands->configure(INHERIT("inputSize"),
                              INHERIT("sampleRate"),
                              INHERIT("log"),
                              INHERIT("normalize"),
                              INHERIT("type"),
                              "frequencyBands", _filterFrequencies,
                              "weighting",_weighting);
}

void MelBands::calculateFilterFrequencies() {
  int filterSize = _numBands;

  _filterFrequencies.resize(filterSize + 2);

  // get the low and high frequency bounds in mel frequency
  Real lowMelFrequencyBound = (*_warper)(parameter("lowFrequencyBound").toReal());
  Real highMelFrequencyBound = (*_warper)(parameter("highFrequencyBound").toReal());
  Real melFrequencyIncrement = (highMelFrequencyBound - lowMelFrequencyBound)/(filterSize + 1);

  Real melFreq = lowMelFrequencyBound;
  for (int i=0; i<filterSize + 2; ++i) {
    _filterFrequencies[i] = (*_inverseWarper)(melFreq);
    melFreq += melFrequencyIncrement; // increment linearly in mel-scale
  }
}


void MelBands::compute() {
  const std::vector<Real>& spectrum = _spectrumInput.get();
  std::vector<Real>& bands = _bandsOutput.get();

  _triangularBands->input("spectrum").set(spectrum);
  _triangularBands->output("bands").set(bands);
  _triangularBands->compute();
}


void MelBands::setWarpingFunctions(std::string warping, std::string weighting){

  if (warping == "htkMel"){
    _warper = hz2mel10;
    _inverseWarper = mel102hz;
  }
  else if (warping == "slaneyMel") {
    _warper = hz2melSlaney;
    _inverseWarper = mel2hzSlaney;
  }
  else{
    E_INFO("Melbands: 'warpingFormula' = "<<warping);
    throw EssentiaException(" Melbands: Bad 'warpingFormula' parameter");
  }

  if (weighting == "warping") {
    _weighting = warping;
  }
  else if (weighting == "linear") {
    _weighting = "linear";
  }
  else{
    throw EssentiaException("Melbands: Bad 'weighting' parameter");
  }

}

