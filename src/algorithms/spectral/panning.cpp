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

#include "panning.h"
#include "essentiamath.h"

using namespace std;
using namespace TNT;
using namespace essentia;
using namespace standard;

const char* Panning::name = "Panning";
const char* Panning::category = "Spectral";
const char* Panning::description = DOC("This algorithm characterizes panorama distribution by comparing spectra from the left and right channels. The panning coefficients are extracted by:\n\n"
"- determining the spatial location of frequency bins given left and right channel spectra;\n\n"
"- computing panorama histogram weighted by the energy of frequency bins, averaging it across frames and normalizing;\n\n"
"- converting the normalized histogram into panning coefficients (IFFT of the log-histogram).\n\n"
"The resulting coefficients will show peaks on the initial bins for left panned audio, and right panning will appear as peaks in the upper bins.\n"
"\n"
"Since panning can vary very rapidly from one frame to the next, the coefficients can be averaged over a time window of several frames by specifying \"averageFrames\" parameter. If a single vector of panning coefficients for the whole audio input is required, \"averageFrames\" should correspond to the length of audio input. In standard mode, sequential runs of compute() method on each frame are required for averaging across frames.\n"
"\n"
"Application: music classification, in particular genre classification [2].\n"
"\n"
"Note: At present time, the original algorithm has not been tested in multi-band mode. That is, numBands must remain 1.\n"
"References:\n"
"  [1] E. Gómez, P. Herrera, P. Cano, J. Janer, J. Serrà, J. Bonada,\n"
"  S. El-Hajj, T. Aussenac, and G. Holmberg, \"Music similarity systems and\n"
"  methods using descriptors,” U.S. Patent WO 2009/0012022009.\n\n"
"  [2] Guaus, E. (2009). Audio content processing for automatic music genre\n"
"  classification: descriptors, databases, and classifiers. PhD Thesis.");


void Panning::correctAudibleAngle(vector<Real>& ratios) {
  Real x = 0;

  for(int i = 0; i < (int)ratios.size(); i++) {
    x = ratios[i];
    if(x < 0.5) {
      x = 1 - x;
      ratios[i] = 1 - (-0.5 + 2.5 * x - 1 * (x * x));
    }
    else {
      ratios[i] = -0.5 + 2.5 * x - 1 * (x * x);
    }
  }
}

void Panning::calculateHistogram(const vector<Real>& subL, const vector<Real>& subR,
                                 vector<Real>& subRatios, vector<Real>& histogram)
{
  //if (histogram.size() != _panningBins) histogram.resize(_panningBins);
  histogram.assign(histogram.size(), 0.);
  int index;

  if(_warpedPanorama) {
    correctAudibleAngle(subRatios);
  }

  for(int i = 0; i < (int)subRatios.size(); i++) {
    index = int(floor(_panningBins * subRatios[i]));
    histogram[index] += subL[i] + subR[i];
  }
}

void Panning::calculateCoefficients(const vector<Real>& histAcum, vector<complex<Real> >& coeffs) {
  int sizeHist = (int)histAcum.size(), i;
  if ((int)coeffs.size() != sizeHist) coeffs.resize(sizeHist);

  Real sumHist = std::accumulate(histAcum.begin(), histAcum.end(), 0.0);
  if (sumHist == 0.0) sumHist = 1.0;

  for (i=0; i<sizeHist; i++) {
    // normalize and take the logarithm
    coeffs[i] = histAcum[i] ? log(histAcum[i]/sumHist) : -230.2585092994046; // log(1.0e-100)
  }
}

void Panning::configure() {
  _averageFrames = parameter("averageFrames").toInt();
  _panningBins = parameter("panningBins").toInt();
  _numCoeffs = parameter("numCoeffs").toInt();
  _numBands = parameter("numBands").toInt();
  _warpedPanorama = parameter("warpedPanorama").toBool();
  _sampleRate = parameter("sampleRate").toReal();
  _histogramAccumulated.resize(_panningBins);
  _ifft->configure("size", _panningBins * 2);
  _nFrames = 0;
}

void Panning::compute() {
  const vector<Real>& spectrumLeft = _spectrumLeft.get();
  const vector<Real>& spectrumRight = _spectrumRight.get();
  Array2D<Real>& panningCoeffs = _panningCoeffs.get();
  // a little story:
  // when panning used to be extractor_panning, the extractor used _frameSize/2
  // as specSize instead of the size of the input spectrum
  // of the input spectrum. Thus:
  // size_t specSize = int(floor(_frameSize*0.5));
  // however, Jordi Janer says that in his original algorithm this was set to
  // frameSize/2, because only half of the spectrum is needed as the other half is
  // symmetric. This is already taken into account in Essentia and for this reason we
  // use the size of the input spectrum. Of course this implementation will give
  // different results as the previous one in extractor_panning.

  // Other differences may appear when comparing the output of this algorithm with
  // the one from extractor_panning. Namely, whether frameCutter or frameCreator is used
  // and the parameters used in the configuration of those.

  if (spectrumLeft.size() != spectrumRight.size()) {
    throw EssentiaException("Panning: spectra for left and right are not of the same size.");
  }
  if (spectrumLeft.empty() || spectrumRight.empty()) {
    throw EssentiaException("Panning: input spectrum empty");
  }
  Real minReal = numeric_limits<Real>::min();
  int specSize = (int) spectrumLeft.size();
  Real fftSize = 2 *_panningBins;
  Real average = _averageFrames ? 1./Real(_averageFrames) : 0;
  vector<Real> histogram(_panningBins);
  vector<Real> ratios(specSize);
  vector<Real> specL(specSize);
  vector<Real> specR(specSize);
  vector<Real> subRatios, subSpecL, subSpecR;
  vector<vector<Real> > panCoeffs(_numBands, vector<Real>(_numCoeffs));

  // Pre-processing
  vector<complex<Real> > inputCoeffs;
  vector<Real> outputCoeffs;
  _ifft->input("fft").set(inputCoeffs);
  _ifft->output("frame").set(outputCoeffs);

  vector<Real> melBands(_numBands + 1, 0.0);
  Real melDiff, melInc, auxTrans;

  melBands[0] = 0.0;
  melBands[_numBands] = _sampleRate*0.5;
  melDiff = hz2mel(melBands[_numBands]) - hz2mel(melBands[0]);
  melInc = melDiff / _numBands;
  auxTrans = 2. * (specSize - 1.0) / _sampleRate;
  melBands[0] *= auxTrans;
  melBands[_numBands] *= auxTrans;

  for (int i = 1; i < _numBands; i++) {
    melBands[i] = mel2hz(i * melInc) * auxTrans;
  }

  // Compute ratios of the magnitudes of both spectra. The result
  // represent the spatial localization of each frequency bin of the spectrum. 
  // The range of the azimuth angle of the panorama is [−45, +45] degrees.
  for (int i = 0; i < specSize; i++) {
    specL[i] = spectrumLeft[i] + minReal;
    specR[i] = spectrumRight[i] + minReal;
    ratios[i] = atan(specR[i] / specL[i]) / (0.5 * M_PI);
  }

  for (int i = 0; i < _numBands; i++) {
    // Group spectra into Mel bands
    int begin =  (int)floor(melBands[i]);
    int n_elems = (int)floor(melBands[i+1] - melBands[i] + 1);
    if (begin + n_elems > specSize) n_elems = specSize - begin;
    vector<Real>::const_iterator it = specL.begin()+begin;
    subSpecL.assign(it, it + n_elems);
    it = specR.begin()+begin;
    subSpecR.assign(it, it+n_elems);
    it = ratios.begin()+begin;
    subRatios.assign(it, it+n_elems);

    // Compute energy weight histogram by weighting each frequency bin 
    // by the energy of spectrum in that bin. 
    calculateHistogram(subSpecL, subSpecR, subRatios, histogram);
  
    if ((_averageFrames == 0) || (_nFrames == 0)) {
      _histogramAccumulated = histogram;
    }
    else {
      // Average histograms over frames
      for (int j = 0; j < (int)histogram.size(); j++) {
        _histogramAccumulated[j] = ((1-average) * _histogramAccumulated[j] + average * histogram[j]);
      }
    }

    // Normalize (unit-sum) to produce an averaged histogram that is independent 
    // of the energy in the audio signal. Take log of the histogram.
    calculateCoefficients(_histogramAccumulated, inputCoeffs);
  
    _ifft->compute();

    for (int j = 0; j < _numCoeffs; j++) {
      //scale coefficients to conform with the results obtained with matlab, as
      //matlab and fftw differ in 1/N where N is the size of the FFT
      panCoeffs[i][j] = outputCoeffs[j] /= fftSize;
    }
  }
  _nFrames++;

  // copy resulting panCoeffs to output:
  if (panningCoeffs.dim1() != (int)panCoeffs.size()) {
    panningCoeffs = Array2D<Real>(panCoeffs.size(), panCoeffs[0].size());
  }
  for (int i = 0; i < (int)panCoeffs.size(); i++) {
    for (int j = 0; j < (int)panCoeffs[i].size(); j++) {
      panningCoeffs[i][j] = panCoeffs[i][j];
    }
  }
}

void Panning::reset() {
  for (int i = 0; i < (int)_histogramAccumulated.size(); i++) {
    _histogramAccumulated[i] = 0.0;
  }
}
