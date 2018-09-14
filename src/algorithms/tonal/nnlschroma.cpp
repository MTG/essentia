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

#include "nnlschroma.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* NNLSChroma::name = "NNLSChroma";
const char* NNLSChroma::category = "Tonal";
const char* NNLSChroma::description = DOC("This algorithm extracts treble and bass chromagrams from a sequence of log-frequency spectrum frames.\n"
"On this representation, two processing steps are performed:\n"
"  -tuning, after which each centre bin (i.e. bin 2, 5, 8, ...) corresponds to a semitone, even if the tuning of the piece deviates from 440 Hz standard pitch.\n"
"  -running standardisation: subtraction of the running mean, division by the running standard deviation. This has a spectral whitening effect.\n"
"This code is ported from NNLS Chroma [1, 2]. To achieve similar results follow this processing chain:\n"
"frame slicing with sample rate = 44100, frame size = 16384, hop size = 2048 -> Windowing with Hann and no normalization -> Spectrum -> LogSpectrum.\n"
"\n"
"References:\n"
"  [1] Mauch, M., & Dixon, S. (2010, August). Approximate Note Transcription\n"
"  for the Improved Identification of Difficult Chords. In ISMIR (pp. 135-140).\n"
"  [2] Chordino and NNLS Chroma,\n"
"  http://www.isophonics.net/nnls-chroma");

const int nBPS = 3; // bins per semitone
const int nOctave = 7;
const int nNote = nOctave * 12 * nBPS + 2 * (nBPS/2+1); // a core over all octaves, plus some overlap at top and bottom

static const Real basswindow[] = {
    0.001769, 0.015848, 0.043608, 0.084265, 0.136670, 0.199341, 0.270509,
    0.348162, 0.430105, 0.514023, 0.597545, 0.678311, 0.754038, 0.822586,
    0.882019, 0.930656, 0.967124, 0.990393, 0.999803, 0.995091, 0.976388,
    0.944223, 0.899505, 0.843498, 0.777785, 0.704222, 0.624888, 0.542025,
    0.457975, 0.375112, 0.295778, 0.222215, 0.156502, 0.100495, 0.055777,
    0.023612, 0.004909, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};
static const Real treblewindow[] = {
    0.000350, 0.003144, 0.008717, 0.017037, 0.028058, 0.041719, 0.057942,
    0.076638, 0.097701, 0.121014, 0.146447, 0.173856, 0.203090, 0.233984,
    0.266366, 0.300054, 0.334860, 0.370590, 0.407044, 0.444018, 0.481304,
    0.518696, 0.555982, 0.592956, 0.629410, 0.665140, 0.699946, 0.733634,
    0.766016, 0.796910, 0.826144, 0.853553, 0.878986, 0.902299, 0.923362,
    0.942058, 0.958281, 0.971942, 0.982963, 0.991283, 0.996856, 0.999650,
    0.999650, 0.996856, 0.991283, 0.982963, 0.971942, 0.958281, 0.942058,
    0.923362, 0.902299, 0.878986, 0.853553, 0.826144, 0.796910, 0.766016,
    0.733634, 0.699946, 0.665140, 0.629410, 0.592956, 0.555982, 0.518696,
    0.481304, 0.444018, 0.407044, 0.370590, 0.334860, 0.300054, 0.266366,
    0.233984, 0.203090, 0.173856, 0.146447, 0.121014, 0.097701, 0.076638,
    0.057942, 0.041719, 0.028058, 0.017037, 0.008717, 0.003144, 0.000350};

void NNLSChroma::configure() {
  _frameSize = parameter("frameSize").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _whitening = parameter("spectralWhitening").toReal();
  _spectralShape = parameter("spectralShape").toReal();
  _useNNLS = parameter("useNNLS").toBool();

  string tuningString = parameter("tuningMode").toString();
  if (tuningString == "local")
    _tuningMode = true;
  if (tuningString == "global")
    _tuningMode = false;

  string normalizationString = parameter("chromaNormalization").toString();
  if (normalizationString == "none")
    _doNormalizeChroma = 0;
  if (normalizationString == "maximum")
    _doNormalizeChroma = 1;
  if (normalizationString == "L1")
    _doNormalizeChroma = 2;
  if (normalizationString == "L2")
    _doNormalizeChroma = 3;


  // Make things for tuning estimation.
  _sinvalues.clear();
  _cosvalues.clear();
	for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
    _sinvalues.push_back(sin(2 * M_PI * (iBPS * 1.0 / nBPS)));
    _cosvalues.push_back(cos(2 * M_PI * (iBPS * 1.0 / nBPS)));
  }

	// Make hamming window of length 1/2 octave.
	int hamwinlength = nBPS * 6 + 1;

  Real hamwinsum = 0;
  _hw.clear();
  for (int i = 0; i < hamwinlength; ++i) { 
    _hw.push_back(0.54 - 0.46 * cos((2 * M_PI * i) / (hamwinlength - 1)));    
    hamwinsum += 0.54 - 0.46 * cos((2 * M_PI * i) / (hamwinlength - 1));
  }

  for (int i = 0; i < hamwinlength; ++i) _hw[i] = _hw[i] / hamwinsum;

  int tempn = nNote * _frameSize;

  vector<Real> tempkernel(tempn);

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

  _dict.assign(nNote * 84, 0.f);
  for (int i = 0; i < nNote * 84; ++i) _dict[i] = 0.0;

  dictionaryMatrix(_dict, _spectralShape);
}

void NNLSChroma::reset() {
  configure();
}

void NNLSChroma::compute() {
  const vector<vector<Real> >& logSpectrum = _logSpectrum.get();
  const vector<Real>& meanTuning = _meanTuning.get();
  const vector<Real>& localTuning = _localTuning.get();
  vector<vector<Real> >& tunedLogfreqSpectrum = _tunedLogfreqSpectrum.get();
  vector<vector<Real> >& semitoneSpectrum = _semitoneSpectrum.get();
  vector<vector<Real> >& bassChromagram = _bassChromagram.get();
  vector<vector<Real> >& chromagram = _chromagram.get();

  if (logSpectrum.size() <= 1)
    throw EssentiaException("NNLSChroma: input vector is empty");

  if (logSpectrum[0].size() != 256) {
    throw EssentiaException("NNLSChroma: log spectrum size should be 256 but it is ", 
                            logSpectrum[0].size(), ".");
  }

  /**  Calculate Tuning
       calculate tuning from (using the angle of the complex number defined by the 
       cumulative mean real and imag values)
  **/
  Real meanTuningImag = 0;
  Real meanTuningReal = 0;
  for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
    meanTuningReal += meanTuning[iBPS] * _cosvalues[iBPS];
    meanTuningImag += meanTuning[iBPS] * _sinvalues[iBPS];
  }

  Real normalisedtuning = atan2(meanTuningImag, meanTuningReal) / (2 * M_PI);
  int intShift = floor(normalisedtuning * 3);
  Real RealShift = normalisedtuning * 3 - intShift; // RealShift is a really bad name for this


  /** Tune Log-Frequency Spectrogram
      calculate a tuned log-frequency spectrogram (tunedLogfreqSpectrum): use the tuning estimated above (kinda f0) to 
      perform linear interpolation on the existing log-frequency spectrogram (kinda f1).
  **/   
  Real tempValue = 0;

  tunedLogfreqSpectrum.assign(logSpectrum.size(), vector<Real>(2, 0.f));
  
  for (int i = 0; i < (int)logSpectrum.size(); i++) {
    if (_tuningMode) {
        intShift = floor(localTuning[i] * 3.f);
        RealShift = localTuning[i] * 3.f - intShift; // RealShift is a really bad name for this
    }
        
    // Interpolate all inner bins.
    for (int k = 2; k < (int)logSpectrum[i].size() - 3; ++k) {
      tempValue = logSpectrum[i][k + intShift] * (1 - RealShift) +
                  logSpectrum[i][k + intShift + 1] * RealShift;
      tunedLogfreqSpectrum[i].push_back(tempValue);
    }

    tunedLogfreqSpectrum[i].push_back(0.0);
    tunedLogfreqSpectrum[i].push_back(0.0);
    tunedLogfreqSpectrum[i].push_back(0.0);  // upper edge

    vector<Real> runningmean = SpecialConvolution(tunedLogfreqSpectrum[i], _hw);
    vector<Real> runningstd;

    // First step: squared values into vector (variance).
    for (int j = 0; j < nNote; j++) {
      runningstd.push_back((tunedLogfreqSpectrum[i][j] - runningmean[j]) *
                           (tunedLogfreqSpectrum[i][j] - runningmean[j]));
    }

    // Second step: convolve.
    runningstd = SpecialConvolution(runningstd, _hw);

    for (int j = 0; j < nNote; j++) {  
      runningstd[j] = sqrt(runningstd[j]); // square root to finally have running std
      if (runningstd[j] > 0) {
        tunedLogfreqSpectrum[i][j] = (tunedLogfreqSpectrum[i][j] - runningmean[j]) > 0 ?
          (tunedLogfreqSpectrum[i][j] - runningmean[j]) / pow(runningstd[j], _whitening) : 0;
      }
      if (tunedLogfreqSpectrum[i][j] < 0) {
        throw EssentiaException("ERROR: negative value in log-frequency spectrum");
      }
    }
  }


  /** Semitone spectrum and chromagrams
      Semitone-spaced log-frequency spectrum derived from the tuned log-freq spectrum above. 
      The spectrum is inferred using a non-negative least squares algorithm.
      Three different kinds of chromagram are calculated, "treble", "bass", and "both" (which 
      means bass and treble stacked onto each other).
  **/
  semitoneSpectrum.assign(logSpectrum.size(), vector<Real>());
  chromagram.assign(logSpectrum.size(), vector<Real>());
  bassChromagram.assign(logSpectrum.size(), vector<Real>());

  for (int i = 0; i < (int)logSpectrum.size(); i++) {
    Real b[nNote];

    bool some_b_greater_zero = false;
    Real sumb = 0;

    for (int j = 0; j < nNote; j++) {
      b[j] = tunedLogfreqSpectrum[i][j];
      sumb += b[j];
      if (b[j] > 0) {
        some_b_greater_zero = true;
      }
    }

    // Here's where the non-negative least squares algorithm calculates the note
    // activation x.
    vector<Real> chroma = vector<Real>(12, 0);
    vector<Real> basschroma = vector<Real>(12, 0);
    Real currval;
    int iSemitone = 0;

    if (some_b_greater_zero) {
      if (!_useNNLS) {
        for (int iNote = nBPS / 2 + 2; iNote < nNote - nBPS / 2;
             iNote += nBPS) {
          currval = 0;
          for (int iBPS = -nBPS / 2; iBPS < nBPS / 2 + 1; ++iBPS) {
            currval += b[iNote + iBPS] * (1 - abs(iBPS * 1.0 / (nBPS / 2 + 1)));
          }

          semitoneSpectrum[i].push_back(currval);
          chroma[iSemitone % 12] += currval * treblewindow[iSemitone];
          basschroma[iSemitone % 12] += currval * basswindow[iSemitone];
          iSemitone++;
        }
      }

      else {
        Real x[84 + 1000];
        for (int j = 1; j < 1084; ++j){
          x[j] = 1.0;
        } 

        vector<int> signifIndex;
        int index = 0;
        sumb /= 84.0;

        for (int iNote = nBPS / 2 + 2; iNote < nNote - nBPS / 2;
             iNote += nBPS) {
          Real currval = 0.f;
          for (int iBPS = -nBPS / 2; iBPS < nBPS / 2 + 1; ++iBPS) {
            currval += b[iNote + iBPS];
          }
          if (currval > 0.f) signifIndex.push_back(index);
          semitoneSpectrum[i].push_back(0.f);  // fill the values, change later
          index++;
        }
        Real rnorm;
        Real w[84 + 1000];
        Real zz[84 + 1000];
        int indx[84 + 1000];
        int mode;
        int dictsize = nNote * signifIndex.size();
        Real* curr_dict = new Real[dictsize];
        for (int iNote = 0; iNote < (int)signifIndex.size(); ++iNote) {
          for (int iBin = 0; iBin < nNote; iBin++) {
            curr_dict[iNote * nNote + iBin] =
                1.0 * _dict[signifIndex[iNote] * nNote + iBin];
          }
        }

        nnls(curr_dict, nNote, nNote, signifIndex.size(), b, x, &rnorm, w, zz,
             indx, &mode);
        delete[] curr_dict;

        for (int iNote = 0; iNote < (int)signifIndex.size(); ++iNote) {
          semitoneSpectrum[i][signifIndex[iNote]] = x[iNote];
          chroma[signifIndex[iNote] % 12] +=
              x[iNote] * treblewindow[signifIndex[iNote]];
          basschroma[signifIndex[iNote] % 12] +=
              x[iNote] * basswindow[signifIndex[iNote]];
        }
      }
    }

    else {
      for (int j = 0; j < 84; ++j) semitoneSpectrum[i].push_back(0);
    }

    chromagram[i] = chroma;
    bassChromagram[i] = basschroma;

    if (_doNormalizeChroma > 0) {
      vector<Real> chromanorm = vector<Real>(3, 0);

      switch (_doNormalizeChroma) {
        case 0:  // should never end up here
          break;
        case 1:
          chromanorm[0] =
              *max_element(chromagram[i].begin(), chromagram[i].end());
          chromanorm[1] =
              *max_element(bassChromagram[i].begin(), bassChromagram[i].end());
          chromanorm[2] = max(chromanorm[0], chromanorm[1]);
          break;
        case 2:
          for (vector<Real>::iterator it = chromagram[i].begin();
               it != chromagram[i].end(); ++it) {
            chromanorm[0] += *it;
          }
          for (vector<Real>::iterator it = bassChromagram[i].begin();
               it != bassChromagram[i].end(); ++it) {
            chromanorm[1] += *it;
          }
          break;
        case 3:
          for (vector<Real>::iterator it = chromagram[i].begin();
               it != chromagram[i].end(); ++it) {
            chromanorm[0] += pow(*it, 2);
          }
          chromanorm[0] = sqrt(chromanorm[0]);
          for (vector<Real>::iterator it = bassChromagram[i].begin();
               it != bassChromagram[i].end(); ++it) {
            chromanorm[1] += pow(*it, 2);
          }
          chromanorm[1] = sqrt(chromanorm[1]);
          chromanorm[2] = sqrt(chromanorm[2]);
          break;
      }
      if (chromanorm[0] > 0) {
        for (int j = 0; j < (int)chromagram[i].size(); j++) {
          chromagram[i][j] /= chromanorm[0];
        }
      }
      if (chromanorm[1] > 0) {
        for (int j = 0; j < (int)bassChromagram[i].size(); j++) {
          bassChromagram[i][j] /= chromanorm[1];
        }
      }
    }
  }
}


/** Special Convolution
    Special convolution is as long as the convolvee, i.e. the first argument. 
	  In the "valid" core part of the convolution it contains the usual convolution 
  	values, but the parts at the beginning (ending) that would normally be 
  	calculated using zero padding simply have the same values as the first 
  	(last) valid convolution bin.
**/
vector<Real> NNLSChroma::SpecialConvolution(vector<Real> convolvee, vector<Real> kernel) {
  Real s;
  int m, n;
  int lenConvolvee = convolvee.size();
  int lenKernel = kernel.size();

  vector<Real> Z(nNote, 0);
  assert(lenKernel % 2 != 0);  // no exception handling !!!

  for (n = lenKernel - 1; n < lenConvolvee; n++) {
    s = 0.0;
    for (m = 0; m < lenKernel; m++) {
      s += convolvee[n - m] * kernel[m];
    }
    Z[n - lenKernel / 2] = s;
  }

  // Fill upper and lower pads.
  for (n = 0; n < lenKernel / 2; n++) {
    Z[n] = Z[lenKernel / 2];
  }
  for (n = lenConvolvee; n < lenConvolvee + lenKernel / 2; n++) {
    Z[n - lenKernel / 2] = Z[lenConvolvee - lenKernel / 2 - 1];
  }  
  return Z;
}


/**
  Calculates a matrix that can be used to linearly map from the magnitude spectrum to a pitch-scale spectrum.
  return this always returns true, which is a bit stupid, really. The main purpose of the function is to change the values in the "matrix" pointed to by *outmatrix
*/
bool NNLSChroma::logFreqMatrix(Real fs, int frameSize, vector<Real> outmatrix) {
  // TODO: rewrite so that everyone understands what is done here.
  // TODO: make this more general, such that it works with all minoctave,
  // maxoctave and whatever nBPS (or check if it already does)

  int binspersemitone = nBPS;
  int minoctave = 0;  // this must be 0
  int maxoctave = 7;  // this must be 7
  int oversampling = 80;

  // Linear frequency vector.
  vector<Real> fft_f;
  for (int i = 0; i < frameSize; ++i) {
    fft_f.push_back(i * (fs * 1.0 / ((frameSize - 1.f) * 2.f)));
  }

  Real fft_width = fs / (frameSize - 1.f);

  // Linear oversampled frequency vector.
  vector<Real> oversampled_f;
  for (int i = 0; i < oversampling * frameSize; ++i) {
    oversampled_f.push_back(
        i * ((fs * 1.0 / ((frameSize - 1.f) * 2.f)) / oversampling));
  }

  // Pitch-spaced frequency vector.
  int minMIDI =
      21 + minoctave * 12 - 1;        // this includes one additional semitone!
  int maxMIDI = 21 + maxoctave * 12;  // this includes one additional semitone!
  vector<Real> cq_f;
  Real oob = 1.0 / binspersemitone;  // one over binspersemitone
  for (int i = minMIDI; i < maxMIDI; ++i) {
    for (int k = 0; k < binspersemitone; ++k) {
      cq_f.push_back(440 * pow(2.0, 0.083333333333 * (i + oob * k - 69)));
    }
  }
  cq_f.push_back(440 * pow(2.0, 0.083333 * (maxMIDI - 69)));

  int nFFT = fft_f.size();

  vector<Real> fft_activation;
  for (int iOS = 0; iOS < 2 * oversampling; ++iOS) {
    Real cosp = cospuls(oversampled_f[iOS], fft_f[1], fft_width);
    fft_activation.push_back(cosp);
  }

  for (int i = 0; i < nFFT * (int)cq_f.size(); ++i) {
    outmatrix[i] = 0.f;
  }

  Real cq_activation;
  for (int iFFT = 1; iFFT < nFFT; ++iFFT) {
    // Find frequency stretch where the oversampled vector can be non-zero
    // (i.e. in a window of width fft_width around the current frequency)
    int curr_start = oversampling * iFFT - oversampling;
    int curr_end = oversampling * iFFT +
                   oversampling;  // don't know if I should add "+1" here
    // oversampled_f[curr_end] << endl;
    for (int iCQ = 0; iCQ < (int)cq_f.size(); ++iCQ) {
      if (cq_f[iCQ] * pow(2.0, 0.084) + fft_width > fft_f[iFFT] &&
          cq_f[iCQ] * pow(2.0, -0.084 * 2) - fft_width < fft_f[iFFT]) {
        // within a generous neighbourhood
        for (int iOS = curr_start; iOS < curr_end; ++iOS) {
          cq_activation =
              pitchCospuls(oversampled_f[iOS], cq_f[iCQ], binspersemitone * 12);
          outmatrix[iFFT + nFFT * iCQ] +=
              cq_activation * fft_activation[iOS - curr_start];
        }
      }
    }
  }
  return true;
}


Real NNLSChroma::cospuls(Real x, Real centre, Real width) {
  Real recipwidth = 1.0/width;
  if (abs(x - centre) <= 0.5 * width) {
    return cos((x-centre)*2*M_PI*recipwidth)*.5+.5;
  }
  return 0.0;
}

Real NNLSChroma::pitchCospuls(Real x, Real centre, int binsperoctave) {
  Real warpedf = -binsperoctave * (log2(centre) - log2(x));
  Real out = cospuls(warpedf, 0.0, 2.0);

  // Now scale to correct for note density.
  Real c = log(2.0)/binsperoctave;
  if (x > 0) {
    out = out / (c * x);
  } 
  else {
    out = 0;
  }

  return out;
}

void NNLSChroma::dictionaryMatrix(vector<Real> dm, Real s_param) {
  // TODO: make this more general, such that it works with all minoctave,
  // maxoctave and even more than one note per semitone
  int binspersemitone = nBPS;
  int minoctave = 0;  // this must be 0
  int maxoctave = 7;  // this must be 7

  // Pitch-spaced frequency vector.
  int minMIDI =
      21 + minoctave * 12 - 1;        // this includes one additional semitone!
  int maxMIDI = 21 + maxoctave * 12;  // this includes one additional semitone!
  vector<Real> cq_f;
  Real oob = 1.0 / binspersemitone;  // one over binspersemitone
  for (int i = minMIDI; i < maxMIDI; ++i) {
    for (int k = 0; k < binspersemitone; ++k) {
      cq_f.push_back(440 * pow(2.0, 0.083333333333 * (i + oob * k - 69)));
    }
  }
  cq_f.push_back(440 * pow(2.0, 0.083333 * (maxMIDI - 69)));

  Real Realbin;
  Real curr_amp;
  // Now for every combination calculate the matrix element.
  for (int iOut = 0; iOut < 12 * (maxoctave - minoctave); ++iOut) {
    for (int iHarm = 1; iHarm <= 20; ++iHarm) {
      Realbin = ((iOut + 1) * binspersemitone + 1) +
                binspersemitone * 12 * log2(iHarm);
      curr_amp = pow(s_param, Real(iHarm - 1));
      for (int iNote = 0; iNote < nNote; ++iNote) {
        if (abs(iNote + 1.0 - Realbin) < 2) {
          dm[iNote + nNote * iOut] +=
              cospuls(iNote + 1.0, Realbin, binspersemitone + 0.0) * curr_amp;
        }
      }
    }
  }
}
