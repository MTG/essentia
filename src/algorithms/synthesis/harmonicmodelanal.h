/*
 * Copyright (C) 2006-2015  Music Technology Group - Universitat Pompeu Fabra
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

#ifndef ESSENTIA_HARMONICMODELANAL_H
#define ESSENTIA_HARMONICMODELANAL_H

#include "algorithm.h"
#include "algorithmfactory.h"
#include <fstream>


namespace essentia {
namespace standard {

class HarmonicModelAnal : public Algorithm {

 protected:

  Input<std::vector<Real> > _frame;
  Input<Real> _pitch;
  Output<std::vector<Real> > _magnitudes;
  Output<std::vector<Real> > _frequencies;
  Output<std::vector<Real> > _phases;
  //Output<std::vector<Real> > _stocenv;

//  int _stocSize;
  Algorithm* _spectrum ;
  Algorithm* _window;
  Algorithm* _fft;
  Algorithm*   _pitchDetect;
  Algorithm* _sineModelAnal;
 /* Algorithm* _sineSubtraction;
  Algorithm* _stochasticModelAnal;*/

 // std::vector<Real> _stocFrameIn; // input frame for the stochaastic analysis algorithm

  Real _sampleRate;
  int _nH ; // number of harmonics
 Real _harmDevSlope;
  std::vector<Real> _lasthfreq;
  bool _useExtPitch;
  


 public:
  HarmonicModelAnal() {
    declareInput(_frame, "frame", "the input frame");
    declareInput(_pitch, "pitch", "external pitch input [Hz]. Optional input.");
    declareOutput(_frequencies, "frequencies", "the frequencies of the sinusoidal peaks [Hz]");
    declareOutput(_magnitudes, "magnitudes", "the magnitudes of the sinusoidal peaks");
    declareOutput(_phases, "phases", "the phases of the sinusoidal peaks");

    _window = AlgorithmFactory::create("Windowing");
    _spectrum =  AlgorithmFactory::create("Spectrum");
    _pitchDetect = AlgorithmFactory::create("PitchYinFFT");
    _fft = AlgorithmFactory::create("FFT");
    _sineModelAnal = AlgorithmFactory::create("SineModelAnal");

  }

  ~HarmonicModelAnal() {

    
  delete _window;
  delete _spectrum;
 delete  _pitchDetect;
  delete _fft;
  delete _sineModelAnal;

  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("hopSize", "the hop size between frames", "[1,inf)", 512);
    declareParameter("fftSize", "the size of the internal FFT size (full spectrum size)", "[1,inf)", 2048);
    declareParameter("maxPeaks", "the maximum number of returned peaks", "[1,inf)", 100);
    declareParameter("maxFrequency", "the maximum frequency of the range to evaluate [Hz]", "(0,inf)", 5000.0);
    declareParameter("minFrequency", "the minimum frequency of the range to evaluate [Hz]", "[0,inf)", 0.0);
    declareParameter("magnitudeThreshold", "peaks below this given threshold are not outputted", "(-inf,inf)", 0.0);
    declareParameter("orderBy", "the ordering type of the outputted peaks (ascending by frequency or descending by magnitude)", "{frequency,magnitude}", "frequency");
    // sinusoidal tracking
    declareParameter("maxnSines", "maximum number of sines per frame", "(0,inf)", 100);
     declareParameter("nHarmonics", "maximum number of harmonics per frame", "(0,inf)", 100);
    declareParameter("harmDevSlope", "slope increase of minimum frequency deviation", "(-inf,inf)", 0.01);
    // external pitch input 
    declareParameter("useExternalPitch", "flag to use the external pitch input","{true,false}", false); // if set to true it gets the external pìtch 
   
  }

  void configure();
  void compute();

  //void updateStocInFrame(const std::vector<Real> frameIn, std::vector<Real> &frameAccumulator);
void harmonicDetection(const std::vector<Real> pfreq, const std::vector<Real> pmag, const std::vector<Real> pphase, const Real f0, const int nH,  std::vector<Real> hfreqp, Real fs, Real harmDevSlope/*=0.01*/,  std::vector<Real> &hfreq,  std::vector<Real> &hmag,  std::vector<Real> &hphase);


  static const char* name;
  static const char* description;



 private:

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class HarmonicModelAnal : public StreamingAlgorithmWrapper {

 protected:
  //Sink<std::vector<std::complex<Real> > > _fft; // input
  Sink<std::vector<Real> > _frame; // input
  Sink<Real> _pitch; // input
  Source<std::vector<Real> > _frequencies;
  Source<std::vector<Real> > _magnitudes;
  Source<std::vector<Real> > _phases;
  

 public:
  HarmonicModelAnal() {
    declareAlgorithm("HarmonicModelAnal");
    declareInput(_frame, TOKEN, "frame");
    declareInput(_pitch, TOKEN, "pitch");
    declareOutput(_frequencies, TOKEN, "frequencies");
    declareOutput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_phases, TOKEN, "phases");
  
  }
};

} // namespace streaming
} // namespace essentia




#endif // ESSENTIA_HARMONICMODELANAL_H
