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

#ifndef ESSENTIA_SINEMODELANAL_H
#define ESSENTIA_SINEMODELANAL_H

#include "algorithm.h"
#include "algorithmfactory.h"



namespace essentia {
namespace standard {

typedef std::pair<int,Real> mypair;

class SineModelAnal : public Algorithm {

 protected:
  Input<std::vector<std::complex<Real> > > _fft;
  Output<std::vector<Real> > _magnitudes;
  Output<std::vector<Real> > _frequencies;
  Output<std::vector<Real> > _phases;
  Algorithm* _peakDetect;
  Algorithm* _cartesianToPolar;

 public:
  SineModelAnal() {
    declareInput(_fft, "fft", "the input frame");
    declareOutput(_frequencies, "frequencies", "the frequencies of the sinusoidal peaks [Hz]");
    declareOutput(_magnitudes, "magnitudes", "the magnitudes of the sinusoidal peaks");
    declareOutput(_phases, "phases", "the phases of the sinusoidal peaks");

    _peakDetect = AlgorithmFactory::create("PeakDetection");
    _cartesianToPolar = AlgorithmFactory::create("CartesianToPolar");

  }


  ~SineModelAnal() {
    delete _peakDetect;
    delete _cartesianToPolar;
  }

  void declareParameters() {
    declareParameter("sampleRate", "the sampling rate of the audio signal [Hz]", "(0,inf)", 44100.);
    declareParameter("maxPeaks", "the maximum number of returned peaks", "[1,inf)", 250);
    declareParameter("maxFrequency", "the maximum frequency of the range to evaluate [Hz]", "(0,inf)", 22050.0);
    declareParameter("minFrequency", "the minimum frequency of the range to evaluate [Hz]", "[0,inf)", 0.0);
    declareParameter("magnitudeThreshold", "peaks below this given threshold are not outputted", "(-inf,inf)", -74.); // same default threshold as in Python implementation
    declareParameter("orderBy", "the ordering type of the outputted peaks (ascending by frequency or descending by magnitude)", "{frequency,magnitude}", "frequency");
    // sinusoidal tracking
    declareParameter("maxnSines", "maximum number of sines per frame", "(0,inf)", 100);
    //declareParameter("minSineDur", "minimum duration of sines in seconds", "(0,inf)", 0.01);
    declareParameter("freqDevOffset", "minimum frequency deviation at 0Hz", "(0,inf)", 20.);
    declareParameter("freqDevSlope", "slope increase of minimum frequency deviation", "(-inf,inf)", 0.01);

  }

  void configure();
  void compute();

  void phaseInterpolation(std::vector<Real> fftphase, std::vector<Real> peakFrequencies, std::vector<Real>& peakPhases);
  void sinusoidalTracking(std::vector<Real>& peakMags, std::vector<Real>& peakFrequencies, std::vector<Real>& peakPhases, const std::vector<Real> tfreq, Real freqDevOffset, Real freqDevSlope,  std::vector<Real> &tmagn, std::vector<Real> &tfreqn, std::vector<Real> &tphasen );
  void cleaningSineTrack();

  std::vector<Real> _lasttpeakFrequency;


  static const char* name;
  static const char* category;
  static const char* description;

 private:
  void sort_indexes(std::vector<int> &idx, const std::vector<Real> &v, bool ascending);
  void copy_vector_from_indexes(std::vector<Real> &out, const std::vector<Real> v, const std::vector<int> idx);
  void copy_int_vector_from_indexes(std::vector<int> &out, const std::vector<int> v, const std::vector<int> idx);
  void erase_vector_from_indexes(std::vector<Real> &v, const std::vector<int> idx);

  // support functions for sort_indexes()
  static bool comparator_up ( const mypair& l, const mypair& r);
  static bool comparator_down ( const mypair& l, const mypair& r);

};

} // namespace standard
} // namespace essentia

#include "streamingalgorithmwrapper.h"

namespace essentia {
namespace streaming {

class SineModelAnal : public StreamingAlgorithmWrapper {

 protected:
  Sink<std::vector<std::complex<Real> > > _fft; // input
  Source<std::vector<Real> > _frequencies;
  Source<std::vector<Real> > _magnitudes;
  Source<std::vector<Real> > _phases;

 public:
  SineModelAnal() {
    declareAlgorithm("SineModelAnal");
    declareInput(_fft, TOKEN, "fft");
    declareOutput(_frequencies, TOKEN, "frequencies");
    declareOutput(_magnitudes, TOKEN, "magnitudes");
    declareOutput(_phases, TOKEN, "phases");
  }
};

} // namespace streaming
} // namespace essentia




#endif // ESSENTIA_SINEMODELANAL_H
