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

#include "logspectrum.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* LogSpectrum::name = "LogSpectrum";
const char* LogSpectrum::category = "Spectral";
const char* LogSpectrum::description = DOC("Computes spectrum with logarithmically distributed frequency bins. "
"This code is a reimplementation of the well known NNLS Chroma based in[1].\n"
"\n"
"note: The algorithm uses moving averages to compute the tuning frequencies so it should be reset before starting "
"to process the frames of a new audio file by calling reset() (or configure())\n"
"\n"
"References:\n"
"  [1] Mauch, M., & Dixon, S. (2010, August). Approximate Note Transcription\n"
"  for the Improved Identification of Difficult Chords. In ISMIR (pp. 135-140).");


const int nBPS = 3; // bins per semitone
const int nOctave = 7;
const int nNote = nOctave * 12 * nBPS + 2 * (nBPS/2+1); // a core over all octaves, plus some overlap at top and bottom
const int MIDI_basenote = 45;

static const Real basswindow[] = {0.001769, 0.015848, 0.043608, 0.084265, 0.136670, 0.199341, 0.270509, 0.348162, 0.430105, 0.514023, 0.597545, 0.678311, 0.754038, 0.822586, 0.882019, 0.930656, 0.967124, 0.990393, 0.999803, 0.995091, 0.976388, 0.944223, 0.899505, 0.843498, 0.777785, 0.704222, 0.624888, 0.542025, 0.457975, 0.375112, 0.295778, 0.222215, 0.156502, 0.100495, 0.055777, 0.023612, 0.004909, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};
static const Real treblewindow[] = {0.000350, 0.003144, 0.008717, 0.017037, 0.028058, 0.041719, 0.057942, 0.076638, 0.097701, 0.121014, 0.146447, 0.173856, 0.203090, 0.233984, 0.266366, 0.300054, 0.334860, 0.370590, 0.407044, 0.444018, 0.481304, 0.518696, 0.555982, 0.592956, 0.629410, 0.665140, 0.699946, 0.733634, 0.766016, 0.796910, 0.826144, 0.853553, 0.878986, 0.902299, 0.923362, 0.942058, 0.958281, 0.971942, 0.982963, 0.991283, 0.996856, 0.999650, 0.999650, 0.996856, 0.991283, 0.982963, 0.971942, 0.958281, 0.942058, 0.923362, 0.902299, 0.878986, 0.853553, 0.826144, 0.796910, 0.766016, 0.733634, 0.699946, 0.665140, 0.629410, 0.592956, 0.555982, 0.518696, 0.481304, 0.444018, 0.407044, 0.370590, 0.334860, 0.300054, 0.266366, 0.233984, 0.203090, 0.173856, 0.146447, 0.121014, 0.097701, 0.076638, 0.057942, 0.041719, 0.028058, 0.017037, 0.008717, 0.003144, 0.000350};



void LogSpectrum::configure() {
  // get parameters
  _frameSize = parameter("frameSize").toInt();
  _sampleRate = parameter("sampleRate").toFloat();
  _rollon = parameter("rollOn").toFloat();
  initialize();
}


void LogSpectrum::compute() {
  const vector<Real>& const_spectrum = _spectrum.get();
  vector<Real> spectrum = const_spectrum;
  vector<Real>& logFreqSpectrum = _logFreqSpectrum.get();
  Real& localTuning = _localTuning.get();
  vector<Real>& meanTuning = _meanTuning.get();

  if (spectrum.size() <= 1)
    throw EssentiaException("LogSpectrum: input vector is empty");

  if (spectrum.size() != _frameSize) {
    E_INFO("LogSpectrum: input spectrum size does not match '_frameSize' parameter. Reconfiguring the algorithm.");
    _frameSize = spectrum.size();
    initialize();
  }

  _frameCount++;   

  Real energysum = 0;

  // make spectrum
  Real maxmag = -10000;
  for (int iBin = 0; iBin < static_cast<int>(_frameSize); iBin++) {
    if (spectrum[iBin]>_frameSize*1.0) spectrum[iBin] = _frameSize;
    if (maxmag < spectrum[iBin]) maxmag = spectrum[iBin];

    if (_rollon > 0) {
      energysum += pow(spectrum[iBin], 2);
    }
  }

  Real cumenergy = 0;
  if (_rollon > 0) {
    for (int iBin = 2; iBin < static_cast<int>(_frameSize); iBin++) {
      cumenergy +=  pow(spectrum[iBin], 2);
      if (cumenergy < energysum * _rollon / 100) spectrum[iBin-2] = 0;
      else break;
    }
  }

  // in the original implementation, "maxmag < 2". In the typical essentia pipe the Window is 
  // normalized to area == 1 making maxmag < 1 most of the time. Thus, this threshold is comented.   
  // if (maxmag < 1.f) {
  //   for (int iBin = 0; iBin < static_cast<int>(_frameSize); iBin++) {
  //     spectrum[iBin] = 0;
  //   }
  // }

  // note spectrum mapping using pre-calculated matrix
  logFreqSpectrum.assign(nNote, 0.f);

  int binCount = 0;
  for (vector<Real>::iterator it = _kernelValue.begin(); it != _kernelValue.end(); ++it) {
    logFreqSpectrum[_kernelNoteIndex[binCount]] += spectrum[_kernelFftIndex[binCount]] * _kernelValue[binCount];
    binCount++;	
  }

  Real one_over_N = 1.0 / _frameCount;

  // update means of complex tuning variables
  for (int iBPS = 0; iBPS < nBPS; ++iBPS) _meanTunings[iBPS] *= float(_frameCount - 1) * one_over_N;

  for (int iTone = 0; iTone < round(nNote*0.62/nBPS)*nBPS+1; iTone = iTone + nBPS) {
    for (int iBPS = 0; iBPS < nBPS; ++iBPS) _meanTunings[iBPS] += logFreqSpectrum[iTone + iBPS] * one_over_N;
    Real ratioOld = 0.997;
    for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
      _localTunings[iBPS] *= ratioOld; 
      _localTunings[iBPS] += logFreqSpectrum[iTone + iBPS] * (1 - ratioOld);
    }
  }

  Real localTuningImag = 0;
  Real localTuningReal = 0;
  for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
    localTuningReal += _localTunings[iBPS] * _cosvalues[iBPS];
    localTuningImag += _localTunings[iBPS] * _sinvalues[iBPS];
  }

  localTuning = atan2(localTuningImag, localTuningReal)/(2 * M_PI);
  meanTuning = _meanTunings;
}


/**
 * Calculates a matrix that can be used to linearly map from the magnitude spectrum to a pitch-scale spectrum.
 * @return this always returns true, which is a bit stupid, really. The main purpose of the function is to change the values in the "matrix" pointed to by *outmatrix
 */
bool LogSpectrum::logFreqMatrix(Real fs, int frameSize, Real *outmatrix) {
	// TODO: rewrite so that everyone understands what is done here.
	// TODO: make this more general, such that it works with all minoctave, maxoctave and whatever nBPS (or check if it already does)

  int binspersemitone = nBPS; 
  int minoctave = 0; // this must be 0
  int maxoctave = 7; // this must be 7
  int oversampling = 80;

  // linear frequency vector
  vector<Real> fft_f;
  for (int i = 0; i < frameSize; ++i) {
    fft_f.push_back(i * (fs * 1.0 / ((frameSize - 1.f ) * 2.f)));
  }

  Real fft_width = fs / (frameSize - 1.f);

  // linear oversampled frequency vector
  vector<Real> oversampled_f;
  for (int i = 0; i < oversampling * frameSize; ++i) {
    oversampled_f.push_back(i * ((fs * 1.0 / ((frameSize - 1.f ) * 2.f)) / oversampling));
  }

  // pitch-spaced frequency vector
  int minMIDI = 21 + minoctave * 12 - 1; // this includes one additional semitone!
  int maxMIDI = 21 + maxoctave * 12; // this includes one additional semitone!
  vector<Real> cq_f;
  Real oob = 1.0/binspersemitone; // one over binspersemitone
  for (int i = minMIDI; i < maxMIDI; ++i) {
    for (int k = 0; k < binspersemitone; ++k)	 {
      cq_f.push_back(440 * pow(2.0,0.083333333333 * (i+oob*k-69)));
    }
  }
  // cq_f.push_back(440 * pow(2.0,0.083333 * (minMIDI-oob-69)));
  cq_f.push_back(440 * pow(2.0,0.083333 * (maxMIDI-69)));

  int nFFT = fft_f.size();

  vector<Real> fft_activation;
  for (int iOS = 0; iOS < 2 * oversampling; ++iOS) {
    Real cosp = cospuls(oversampled_f[iOS],fft_f[1],fft_width);
    fft_activation.push_back(cosp);
  }

  for (int i = 0; i < nFFT * (int)cq_f.size(); ++i) {
    outmatrix[i] = 0.f;
  }

  Real cq_activation;
  for (int iFFT = 1; iFFT < nFFT; ++iFFT) {
    // find frequency stretch where the oversampled vector can be non-zero (i.e. in a window of width fft_width around the current frequency)
    int curr_start = oversampling * iFFT - oversampling;
    int curr_end = oversampling * iFFT + oversampling; // don't know if I should add "+1" here
    for (int iCQ = 0; iCQ < (int)cq_f.size(); ++iCQ) {
      if (cq_f[iCQ] * pow(2.0, 0.084) + fft_width > fft_f[iFFT] && cq_f[iCQ] * pow(2.0, -0.084 * 2) - fft_width < fft_f[iFFT]) { // within a generous neighbourhood
        for (int iOS = curr_start; iOS < curr_end; ++iOS) {
          cq_activation = pitchCospuls(oversampled_f[iOS],cq_f[iCQ],binspersemitone*12);
          outmatrix[iFFT + nFFT * iCQ] += cq_activation * fft_activation[iOS-curr_start];
        }
      }
    }
  }
  return true;	
}

Real LogSpectrum::cospuls(Real x, Real centre, Real width) {
  Real recipwidth = 1.0/width;
  if (abs(x - centre) <= 0.5 * width) {
    return cos((x-centre)*2*M_PI*recipwidth)*.5+.5;
  }
  return 0.0;
}

Real LogSpectrum::pitchCospuls(Real x, Real centre, int binsperoctave) {
  Real warpedf = -binsperoctave * (log2(centre) - log2(x));
  Real out = cospuls(warpedf, 0.0, 2.0);

  // now scale to correct for note density
  Real c = log(2.0)/binsperoctave;
  if (x > 0) {
    out = out / (c * x);
  } 
  else {
    out = 0;
  }

  return out;
}

void LogSpectrum::initialize() {
	// make things for tuning estimation
  _sinvalues.clear();
  _cosvalues.clear();
	for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
    _sinvalues.push_back(sin(2 * M_PI * (iBPS * 1.0 / nBPS)));
    _cosvalues.push_back(cos(2 * M_PI * (iBPS * 1.0 / nBPS)));
  }

  _localTunings.clear();
  _meanTunings.clear();
  for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
    _meanTunings.push_back(0);
    _localTunings.push_back(0);
  }

  _frameCount = 0;

  int tempn = nNote * _frameSize;

  Real *tempkernel;
  tempkernel = new Real[tempn];

  logFreqMatrix(_sampleRate, _frameSize, tempkernel);
  _kernelValue.clear();
  _kernelFftIndex.clear();
  _kernelNoteIndex.clear();
  int countNonzero = 0;
  for (int iNote = 0; iNote < nNote; ++iNote) {
    for (int iFFT = 0; iFFT <static_cast<int>(_frameSize); ++iFFT) {
      if (tempkernel[iFFT + _frameSize * iNote] > 0) {
        _kernelValue.push_back(tempkernel[iFFT + _frameSize * iNote]);
        if (tempkernel[iFFT + _frameSize * iNote] > 0) {
            countNonzero++;
        }
        _kernelFftIndex.push_back(iFFT);
        _kernelNoteIndex.push_back(iNote);				
      }
    }
  }
  delete [] tempkernel;
}

void LogSpectrum::reset() {
  initialize();
}
