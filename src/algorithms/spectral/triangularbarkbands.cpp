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

#include "triangularbarkbands.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* TriangularBarkBands::name = "TriangularBarkBands";
const char* TriangularBarkBands::category = "Spectral";
const char* TriangularBarkBands::description = DOC("This algorithm computes energy in mel bands of a spectrum. It applies a frequency-domain filterbank (MFCC FB-40, [1]), which consists of equal area triangular filters spaced according to the mel scale. The filterbank is normalized in such a way that the sum of coefficients for every filter equals one. It is recommended that the input \"spectrum\" be calculated by the Spectrum algorithm.\n"
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
"  http://en.wikipedia.org/wiki/Mel_frequency_cepstral_coefficient");

void TriangularBarkBands::configure() {
  if (parameter("highFrequencyBound").toReal() > parameter("sampleRate").toReal()*0.5 ) {
    throw EssentiaException("TriangularBarkBands: High frequency bound cannot be higher than Nyquist frequency");
  }
  if (parameter("highFrequencyBound").toReal() <= parameter("lowFrequencyBound").toReal()) {
    throw EssentiaException("TriangularBarkBands: High frequency bound cannot be lower than the low frequency bound.");
  }
  _numBands = parameter("numberBands").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _normalization = parameter("normalize").toString();
  _type = parameter("type").toString();
    
    _isLog = parameter("log").toBool();
    calculateFilterCoefficients();
}

void TriangularBarkBands::calculateFilterCoefficients() {
    int nfft = (parameter("inputSize").toInt()-1)*2;
    int nfilts = _numBands;
    int sr = _sampleRate;
    float width = 1.0;
    
    float minfreq = parameter("lowFrequencyBound").toReal();
    float maxfreq = parameter("highFrequencyBound").toReal();
    
    float min_bark = _hz2bark(minfreq);
    float nyqbark = _hz2bark(maxfreq) - min_bark;
    
    if(nfilts == 0)
        nfilts = ceil(nyqbark)+1;
    
    _filterCoefficients.resize(nfilts);
    
    float step_barks = nyqbark/(nfilts-1);
    
    std::vector<Real> binbarks;
    
    float srOverNFFT = (float)sr/nfft;
    
    for(int i=0; i<nfft/2+1; i++)
        binbarks.push_back(_hz2bark((float)i*srOverNFFT));
    
    for(int i=0; i<nfilts; i++)
        _filterCoefficients[i].resize(binbarks.size());
    
    for(int i = 0; i < nfilts; i++)
    {
        float f_bark_mid = min_bark + i * step_barks;
        
        for(int j=0; j<binbarks.size(); j++)
        {
            float lof = binbarks[j] - f_bark_mid - 0.5;
            float hif = binbarks[j] - f_bark_mid + 0.5;
            
            double coeff = std::min((float)0, min((float)hif, (float)-2.5*lof)/width);
            
            _filterCoefficients[i][j] = pow(10, coeff);
        }
    }
    
    // normalize the filter weights
    if ( _normalization.compare("unit_sum") == 0 ){
        for (int i=0; i<nfilts; ++i) {
            Real weight = 0.0;
            
            for (int j=0; j<binbarks.size(); ++j) {
                weight += _filterCoefficients[i][j];
            }
            
            if (weight == 0) continue;
            
            for (int j=0; j<binbarks.size(); ++j) {
                _filterCoefficients[i][j] = _filterCoefficients[i][j] / weight;
            }
        }
    }    
}


void TriangularBarkBands::compute() {
  const std::vector<Real>& spectrum = _spectrumInput.get();
  std::vector<Real>& bands = _bandsOutput.get();
    
    if (spectrum.size() <= 1) {
        throw EssentiaException("TriangularBands: the size of the input spectrum is not greater than one");
    }
    
    int filterSize = _numBands;
    int spectrumSize = spectrum.size();
    
    bands.resize(_numBands);
    fill(bands.begin(), bands.end(), (Real) 0.0);
    
    for (int i=0; i<filterSize; ++i) {
        for (int j=0; j<spectrum.size(); ++j) {
            if (_type == "power"){
                bands[i] += (spectrum[j] * spectrum[j]) * _filterCoefficients[i][j];
            }
            
            if (_type == "magnitude"){
                bands[i] += (spectrum[j]) * _filterCoefficients[i][j];
            }
            
            if (_isLog)
                bands[i] = log2(1 + bands[i]);
        }
    }
}

