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

#include "tonicindianartmusic.h"
#include "essentiamath.h"

using namespace std;


namespace essentia {
namespace standard {


const char* TonicIndianArtMusic::name = "TonicIndianArtMusic";
const char* TonicIndianArtMusic::version = "1.0";
const char* TonicIndianArtMusic::description = DOC("This algorithm estimates the tonic frequency of the lead artist in Indian art music. It uses multipitch representation of the audio signal (pitch salience) to compute a histogram using which the tonic is identified as one of its peak. The decision is made based on the distance between the prominent peaks, the classification is done using a decision tree.\n"
"\n"
"References:\n"
"  [1] J. Salamon, S. Gulati, and X. Serra, \"A Multipitch Approach to Tonic\n"
"  Identification in Indian Classical Music,\" in International Society for\n"
"  Music Information Retrieval Conference (ISMIR’12), 2012.");


void TonicIndianArtMusic::configure() {

  Real sampleRate = parameter("sampleRate").toReal();
  int frameSize = parameter("frameSize").toInt();
  int hopSize = parameter("hopSize").toInt();
  string windowType = "hann";
  int zeroPaddingFactor = 4;
  int maxSpectralPeaks = 100;
  int numberHarmonics = parameter("numberHarmonics").toInt();
  Real harmonicWeight = parameter("harmonicWeight").toReal();
  Real magnitudeThreshold = parameter("magnitudeThreshold").toReal();
  Real magnitudeCompression = parameter("magnitudeCompression").toReal();
  Real minTonicFrequency = parameter("minTonicFrequency").toReal();
  Real maxTonicFrequency = parameter("maxTonicFrequency").toReal();

  _referenceFrequency = parameter("referenceFrequency").toReal();
  _binResolution = parameter("binResolution").toReal();
  _numberSaliencePeaks = parameter("numberSaliencePeaks").toReal();
  _numberBins = floor(6000.0 / _binResolution) - 1;

  // Pre-processing
  _frameCutter->configure("frameSize", frameSize,
                           "hopSize", hopSize);

  _windowing->configure("size", frameSize,
                        "zeroPadding", (zeroPaddingFactor-1) * frameSize,
                        "type", windowType);
  // Spectral peaks
  _spectrum->configure("size", frameSize * zeroPaddingFactor);

  // TODO which value to select for maxFrequency? frequencies up to 1.76kHz * numHarmonics will
  //      theoretically affect the salience function computation

  _spectralPeaks->configure(
                            "minFrequency", 1,  // to avoid zero frequencies
                            "maxFrequency", 20000,
                            "maxPeaks", maxSpectralPeaks,
                            "sampleRate", sampleRate,
                            "magnitudeThreshold", 0,
                            "orderBy", "magnitude");

  // Pitch salience contours
  _pitchSalienceFunction->configure("binResolution", _binResolution,
                                    "referenceFrequency", _referenceFrequency,
                                    "magnitudeThreshold", magnitudeThreshold,
                                    "magnitudeCompression", magnitudeCompression,
                                    "numberHarmonics", numberHarmonics,
                                    "harmonicWeight", harmonicWeight);

  _pitchSalienceFunctionPeaks->configure("binResolution", _binResolution,
                                         "minFrequency", minTonicFrequency,
                                         "maxFrequency", maxTonicFrequency,
                                         "referenceFrequency", _referenceFrequency);
}


void TonicIndianArtMusic::compute() {
  const vector<Real>& signal = _signal.get();
  Real& tonic = _tonic.get();

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

  // histogram computation
  vector<Real> histogram;
  histogram.resize(_numberBins);

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

    // calculate salience function
    _pitchSalienceFunction->compute();

    // calculate peaks of salience function
    _pitchSalienceFunctionPeaks->compute();

    // consider only those frames where the number of peaks detected are more than minimum peaks needed for histogram
    if(frameSalienceBins.size()>=_numberSaliencePeaks){
      for (size_t i=0; i<_numberSaliencePeaks; i++) {
	histogram[frameSalienceBins[i]]+=1;
      }
    }
  }

  // computing the peaks of the histogram function
  vector <Real> peak_locs;
  vector <Real> peak_amps;
  Real tonic_loc;

    // configure algorithms [#5 peaks]
  _peakDetection->configure("interpolate", false);
  _peakDetection->configure("range", (int)histogram.size());
  _peakDetection->configure("maxPosition", (int)histogram.size());
  _peakDetection->configure("minPosition", 0);
  _peakDetection->configure("maxPeaks", 5);
  _peakDetection->configure("orderBy", "amplitude");

    // find salience function peaks
  _peakDetection->input("array").set(histogram);
  _peakDetection->output("positions").set(peak_locs);
  _peakDetection->output("amplitudes").set(peak_amps);
  _peakDetection->compute();

  // this is the decision tree hardcoded to choose the peak in the histogram which corresponds o the tonic
  /*implementing the decision tree*/
  Real highest_peak_loc = peak_locs[0];
  Real f2 =peak_locs[1] - highest_peak_loc;
  Real f3 =peak_locs[2] - highest_peak_loc;
  Real f5 =peak_locs[4] - highest_peak_loc;

  if (f2>50){
    tonic_loc = peak_locs[0];
  }
  else{
    if(f2<=-70){
        if(f3<=50){
          tonic_loc = peak_locs[1];
        }
      else{
        tonic_loc = peak_locs[0];
      }
    }
    else{
      if(f3<=-60){
        tonic_loc = peak_locs[2];
      }
      else{
        if(f5<=-80){
          tonic_loc = peak_locs[4];
        }
        else{
          if(f5<=30){
            if(f2<=20){
              tonic_loc = peak_locs[1];
            }
            else{
              tonic_loc = peak_locs[3];
            }
          }
          else{
            tonic_loc = peak_locs[0];
          }
        }
      }
    }
  }
  //converting value of the tonic in cent to Hz scale
  Real _centToHertzBase = pow(2, _binResolution / 1200.0);
  tonic = _referenceFrequency * pow(_centToHertzBase, tonic_loc);
}

TonicIndianArtMusic::~TonicIndianArtMusic() {
    // Pre-processing
    delete _frameCutter;
    delete _windowing;

    // Spectral peaks
    delete _spectrum;
    delete _spectralPeaks;

    // Pitch salience contours
    delete _pitchSalienceFunction;
    delete _pitchSalienceFunctionPeaks;
}


} // namespace standard
} // namespace essentia
