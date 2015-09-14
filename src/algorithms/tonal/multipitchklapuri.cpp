/*
 * Copyright (C) 2006-2013  Music Technology Group - Universitat Pompeu Fabra
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

#include "multipitchklapuri.h"
#include "essentiamath.h"

using namespace std;

namespace essentia {
namespace standard {

const char* MultiPitchKlapuri::name = "MultiPitchKlapuri";
const char* MultiPitchKlapuri::version = "1.0";
const char* MultiPitchKlapuri::description = DOC("This algorithm estimates multiple pitch values corresponding to the melodic lines present in a polyphonic music signal (i.e. string quartet, piano). This implementation is based on the algorithm in [1]: In each frame, a set of possible fundamental frequency candidates is extracted based on the principle of harmonic summation. In an optimization stage, the number of harmonic sources (polyphony) is estimated and the final set of fundamental frequencies determined. In contrast to the pich salience function proposed in [2], this implementation uses the pitch salience function described in [1].
"\n"
"The output is a vector for each frame containing the estimated melody pitch values .\n"
"\n"
"References:\n"
"  [1] A. Klapuri, \"Multiple Fundamental Frequency Estimation by Summing Harmonic\n"
"  Amplitudes \", International Society for Music Information Retrieval Conference\n"
"  (2006).\n"
"  [2] J. Salamon and E. Gómez, \"Melody extraction from polyphonic music\n"
"  signals using pitch contour characteristics,\" IEEE Transactions on Audio,\n"
"  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.\n\n"
);

void MultiPitchKlapuri::configure() {

  sampleRate = parameter("sampleRate").toReal();
  frameSize = parameter("frameSize").toInt();
  hopSize = parameter("hopSize").toInt();
  string windowType = "hann";
  zeroPaddingFactor = 4;
  int maxSpectralPeaks = 100;
  numberBins = frequencyToCentBin(sampleRate/2);
  centSpectrum.resize(numberBins);
  
  referenceFrequency        = parameter("referenceFrequency").toReal();
  binResolution             = parameter("binResolution").toReal();
  Real magnitudeThreshold   = parameter("magnitudeThreshold").toReal();
  Real magnitudeCompression = parameter("magnitudeCompression").toReal();
  numberHarmonics           = parameter("numberHarmonics").toInt();
  Real harmonicWeight       = parameter("harmonicWeight").toReal();
  Real minFrequency         = parameter("minFrequency").toReal();
  Real maxFrequency         = parameter("maxFrequency").toReal();

  numberHarmonicsMax        = floor(sampleRate / maxFrequency);
  numberHarmonicsMax        = min(numberHarmonics, numberHarmonicsMax);
  binsInSemitone            = floor(100.0 / binResolution);
  centToHertzBase           = pow(2, binResolution / 1200.0);
  binsInOctave              = 1200.0 / binResolution;
  referenceTerm             = 0.5 - binsInOctave * log2(referenceFrequency);

  // Pre-processing
  _frameCutter->configure("frameSize", frameSize,
                          "hopSize", hopSize,
                          "startFromZero", false);

  _windowing->configure("size", frameSize,
                        "zeroPadding", (zeroPaddingFactor-1) * frameSize,
                        "type", windowType);
  // Spectral peaks
  _spectrum->configure("size", frameSize * zeroPaddingFactor);


  _spectralPeaks->configure(
              "minFrequency", 1,  // to avoid zero frequencies
              "maxFrequency", 20000,
              "maxPeaks", maxSpectralPeaks,
              "sampleRate", sampleRate,
              "magnitudeThreshold", 0,
              "orderBy", "magnitude");
  

  // Spectral whitening
  _spectralWhitening->configure("sampleRate", sampleRate);
  
  // Pitch salience contours
  _pitchSalienceFunction->configure("binResolution", binResolution,
                  "referenceFrequency", referenceFrequency,
                  "magnitudeThreshold", magnitudeThreshold,
                  "magnitudeCompression", magnitudeCompression,
                  "numberHarmonics", numberHarmonics,
                  "harmonicWeight", harmonicWeight);

  // pitch salience function peaks are considered F0 cadidates -> limit to considered frequency range
  _pitchSalienceFunctionPeaks->configure("binResolution", binResolution,
                     "referenceFrequency", referenceFrequency,
                     "minFrequency", minFrequency,
                     "maxFrequency", maxFrequency);
}

void MultiPitchKlapuri::compute() {
  const vector<Real>& signal = _signal.get();
  vector<vector<Real> >& pitch = _pitch.get();
  if (signal.empty()) {
    pitch.clear();
    return;
  }

  // Pre-processing
  vector<Real> frame;
  _frameCutter->input("signal").set(signal);
  _frameCutter->output("frame").set(frame);

  vector<Real> frameWindowed;
  _windowing->input("frame").set(frame);
  _windowing->output("frame").set(frameWindowed);

  // Spectral peaks
  vector<Real> frameSpectrum;
  _spectrum->input("frame").set(frameWindowed);
  _spectrum->output("spectrum").set(frameSpectrum);

  vector<Real> frameFrequencies;
  vector<Real> frameMagnitudes;
  _spectralPeaks->input("spectrum").set(frameSpectrum);
  _spectralPeaks->output("frequencies").set(frameFrequencies);
  _spectralPeaks->output("magnitudes").set(frameMagnitudes);

  // Spectral whitening
  vector<Real> frameWhiteMagnitudes;
  _spectralWhitening->input("spectrum").set(frameSpectrum);
  _spectralWhitening->input("frequencies").set(frameFrequencies);
  _spectralWhitening->input("magnitudes").set(frameMagnitudes);
  _spectralWhitening->output("magnitudes").set(frameWhiteMagnitudes);
  
  // Pitch salience contours
  vector<Real> frameSalience;
  _pitchSalienceFunction->input("frequencies").set(frameFrequencies);
  _pitchSalienceFunction->input("magnitudes").set(frameMagnitudes);
  _pitchSalienceFunction->output("salienceFunction").set(frameSalience);

  vector<Real> frameSalienceBins;
  vector<Real> frameSalienceValues;
  _pitchSalienceFunctionPeaks->input("salienceFunction").set(frameSalience);
  _pitchSalienceFunctionPeaks->output("salienceBins").set(frameSalienceBins);
  _pitchSalienceFunctionPeaks->output("salienceValues").set(frameSalienceValues);

  vector<Real> nearestBinWeights;
  nearestBinWeights.resize(binsInSemitone + 1);
  for (int b=0; b <= binsInSemitone; b++) {
    nearestBinWeights[b] = pow(cos((Real(b)/binsInSemitone)* M_PI/2), 2);
  }
  
  vector<Real> harmonicWeights;
  harmonicWeights.clear();
  harmonicWeights.reserve(numberHarmonicsMax);
  for (int h=0; h<numberHarmonicsMax; h++) {
    harmonicWeights.push_back(pow(0.8, h));
  }
  
  while (true) {
    // get a frame
    _frameCutter->compute();

    if (!frame.size()) {
      break;
    }

    _windowing->compute();

    // calculate spectrum
    _spectrum->compute();

    // calculate spectral peaks
    _spectralPeaks->compute();
    
    // whiten the spectrum
    _spectralWhitening->compute();

    // calculate salience function
    _pitchSalienceFunction->compute();

    // calculate peaks of salience function
    _pitchSalienceFunctionPeaks->compute();
      
    // no peaks in this frame
    if (!frameSalienceBins.size()) {
      continue;
    }

    // Joint F0 estimation (pitch salience function peaks as candidates) 

    // compute the cent-scaled spectrum
    fill(centSpectrum.begin(), centSpectrum.end(), (Real) 0.0);
    for (int i=0; i<frameSpectrum.size(); i++) {
      Real f = (Real(i) / Real(frameSpectrum.size())) * (sampleRate/2);
      int k = frequencyToCentBin(f);
      if (k>0 && k<numberBins) {
        centSpectrum[k] = centSpectrum[k] + frameSpectrum[i];
      }
    }
  
    // get indices corresponding to harmonics of each found peak
    vector<vector<int> > kPeaks;
    for (int i=0; i<frameSalienceBins.size(); i++) {
      vector<int> k;
      Real f = referenceFrequency * pow(centToHertzBase, frameSalienceBins[i]);
      for (int m=0; m<numberHarmonicsMax; m++) {
        // find the exact peak for each harmonic
        int kBin = frequencyToCentBin(f*(m+1));
        int kBinMin = max(0, int(kBin-binsInSemitone));
        int kBinMax = min(numberBins-1, int(kBin+binsInSemitone));
        vector<Real> specSegment;
        for (int ii=kBinMin; ii<=kBinMax; ii++) {
          specSegment.push_back(centSpectrum[ii]);
        }
        kBin = kBinMin + argmax(specSegment)-1;
        k.push_back(kBin);
      }
      kPeaks.push_back(k);
    }
    
    // candidate Spectra
    vector<vector<Real> > Z;
    for (int i=0; i<frameSalienceBins.size(); i++) {
      vector<Real> z(numberBins, 0.);
      for (int h=0; h<numberHarmonicsMax; h++) {
        int hBin = kPeaks[i][h];
        for(int b=max(0, hBin-binsInSemitone); b <= min(numberBins-1, hBin+binsInSemitone); b++) {
          z[b] += nearestBinWeights[abs(b-hBin)] * getWeight(hBin,h) * 0.25; // 0.25 is cancellation parameter
        }
      }
      Z.push_back(z);
    }

    // TODO: segfault somewhere here
    // inhibition function
    int numCandidates = frameSalienceBins.size();
    vector<vector<Real> > inhibition;

    for (int i=0; i<numCandidates; i++) {
      vector<Real> inh(numCandidates, 0.); 
      for (int j=0; j<numCandidates; j++) {
        for (int m=0; m<numberHarmonicsMax; m++) {
          inh[j] += getWeight(kPeaks[i][m], m) * centSpectrum[kPeaks[i][m]] * Z[j][kPeaks[i][m]];
        }
      }
      inhibition.push_back(inh);
    }

    // polyphony estimation initialization
    vector<int> finalSelection;
    int p = 1;
    Real gamma = 0.73;
    Real S = frameSalienceValues[argmax(frameSalienceValues)] / pow(p,gamma);
    finalSelection.push_back(argmax(frameSalienceValues));
    // goodness function
    vector<vector<Real> > G;
    for (int i=0; i<numCandidates; i++) {
      vector<Real> g;
      for (int j=0; j<numCandidates; j++) {
        if(i==j) {
          g.push_back(0.0);
        } else {
          Real g_val = frameSalienceValues[i] + frameSalienceValues[j] - (inhibition[i][j] + inhibition[j][i]);
          g.push_back(g_val);
        }
      }
      G.push_back(g);
    }
  
    vector<vector<int> > selCandInd;
    vector<Real> selCandVal;
    vector<Real> localF0;
    
    while (true) {
      // find numCandidates largest values
      Real maxVal=-1;
      int maxInd_i=0;
      int maxInd_j=0;
  
      for (int I=0; I<numCandidates; I++) {
        vector<int> localInd;
        for (int i=0; i<numCandidates; i++) {
          for (int j=0; j<numCandidates; j++) {
            if (G[i][j]>maxVal) {
              maxVal=G[i][j];
              maxInd_i=i;
              maxInd_j=j;
            }
          }
        }

        localInd.push_back(maxInd_i);
        localInd.push_back(maxInd_j);
        selCandInd.push_back(localInd);
        selCandVal.push_back(G[maxInd_i][maxInd_j]);
        G[maxInd_i][maxInd_j]=-1;
        maxVal=-1;
        maxInd_i=0;
        maxInd_j=0;
      }
  
      // re-estimate polyphony
      p++;
      Real Snew = selCandVal[argmax(selCandVal)] / pow(p,gamma);
      if (Snew>S) {
        finalSelection.clear();
        for (int i=0; i<selCandInd[0].size(); i++) {
          finalSelection.push_back(selCandInd[0][i]);
        }
        // re-calculate goddess function
        for (int i=0; i<numCandidates; i++) {
          for (int j=0; j<numCandidates; j++) {
            G[i][j]+=frameSalienceValues[j];
            for (int ii=0; ii<selCandInd[i].size(); ii++) {
              G[i][j]-=(inhibition[selCandInd[i][ii]][j]+inhibition[j][selCandInd[i][ii]]);
            }
          }
        }
        S = Snew;
      } 
      else {
        // add estimated f0 to frame
        for (int i=0; i<finalSelection.size(); i++) {
          Real freq=referenceFrequency * pow(centToHertzBase, frameSalienceBins[finalSelection[i]]);
          localF0.push_back(freq);
        }
        break;
      }
    }
    pitch.push_back(localF0);
  }
}

int MultiPitchKlapuri::frequencyToCentBin(Real frequency) {
  // +0.5 term is used instead of +1 (as in [1]) to center 0th bin to 55Hz
  // formula: floor(1200 * log2(frequency / _referenceFrequency) / _binResolution + 0.5)
  //  --> 1200 * (log2(frequency) - log2(_referenceFrequency)) / _binResolution + 0.5
  //  --> 1200 * log2(frequency) / _binResolution + (0.5 - 1200 * log2(_referenceFrequency) / _binResolution)
  return floor(binsInOctave * log2(frequency) + referenceTerm);
}

Real MultiPitchKlapuri::getWeight(int centBin, int harmonicNumber) {
  Real f = referenceFrequency * pow(centToHertzBase, centBin);
  Real alpha = 27.0;
  Real beta = 320.0;
  Real w = (f+alpha) / (harmonicNumber*f+beta);
  return w;
}
  
MultiPitchKlapuri::~MultiPitchKlapuri() {
  // Pre-processing
  delete _frameCutter;
  delete _windowing;

  // Spectral peaks
  delete _spectrum;
  delete _spectralPeaks;
  
  // Spectral whitening
  delete _spectralWhitening;

  // Pitch salience contours
  delete _pitchSalienceFunction;
  delete _pitchSalienceFunctionPeaks;
}

} // namespace standard
} // namespace essentia


