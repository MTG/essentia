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

#include "nnls.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* NNLS::name = "NNLS";
const char* NNLS::category = "Tonal";
const char* NNLS::description = DOC("");

const int nBPS = 3; // bins per semitone
const int nOctave = 7;
const int nNote = nOctave * 12 * nBPS + 2 * (nBPS/2+1); // a core over all octaves, plus some overlap at top and bottom
const int MIDI_basenote = 45;

static const Real basswindow[] = {0.001769, 0.015848, 0.043608, 0.084265, 0.136670, 0.199341, 0.270509, 0.348162, 0.430105, 0.514023, 0.597545, 0.678311, 0.754038, 0.822586, 0.882019, 0.930656, 0.967124, 0.990393, 0.999803, 0.995091, 0.976388, 0.944223, 0.899505, 0.843498, 0.777785, 0.704222, 0.624888, 0.542025, 0.457975, 0.375112, 0.295778, 0.222215, 0.156502, 0.100495, 0.055777, 0.023612, 0.004909, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};
static const Real treblewindow[] = {0.000350, 0.003144, 0.008717, 0.017037, 0.028058, 0.041719, 0.057942, 0.076638, 0.097701, 0.121014, 0.146447, 0.173856, 0.203090, 0.233984, 0.266366, 0.300054, 0.334860, 0.370590, 0.407044, 0.444018, 0.481304, 0.518696, 0.555982, 0.592956, 0.629410, 0.665140, 0.699946, 0.733634, 0.766016, 0.796910, 0.826144, 0.853553, 0.878986, 0.902299, 0.923362, 0.942058, 0.958281, 0.971942, 0.982963, 0.991283, 0.996856, 0.999650, 0.999650, 0.996856, 0.991283, 0.982963, 0.971942, 0.958281, 0.942058, 0.923362, 0.902299, 0.878986, 0.853553, 0.826144, 0.796910, 0.766016, 0.733634, 0.699946, 0.665140, 0.629410, 0.592956, 0.555982, 0.518696, 0.481304, 0.444018, 0.407044, 0.370590, 0.334860, 0.300054, 0.266366, 0.233984, 0.203090, 0.173856, 0.146447, 0.121014, 0.097701, 0.076638, 0.057942, 0.041719, 0.028058, 0.017037, 0.008717, 0.003144, 0.000350};


void NNLS::configure() {
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

  string normalizationString = parameter("chromaNormalisation").toString();
  if (normalizationString == "none")
    _doNormalizeChroma = 0;
  if (normalizationString == "maximum")
    _doNormalizeChroma = 1;
  if (normalizationString == "L1")
    _doNormalizeChroma = 2;
  if (normalizationString == "L2")
    _doNormalizeChroma = 3;


  // make things for tuning estimation
  _sinvalues.clear();
  _cosvalues.clear();
	for (int iBPS = 0; iBPS < nBPS; ++iBPS) {
    _sinvalues.push_back(sin(2 * M_PI * (iBPS * 1.0 / nBPS)));
    _cosvalues.push_back(cos(2 * M_PI * (iBPS * 1.0 / nBPS)));
  }

	// make hamming window of length 1/2 octave
	int hamwinlength = nBPS * 6 + 1;

  Real hamwinsum = 0;
  _hw.clear();
  for (int i = 0; i < hamwinlength; ++i) { 
    _hw.push_back(0.54 - 0.46 * cos((2 * M_PI * i) / (hamwinlength - 1)));    
    hamwinsum += 0.54 - 0.46 * cos((2 * M_PI * i) / (hamwinlength - 1));
  }

  for (int i = 0; i < hamwinlength; ++i) _hw[i] = _hw[i] / hamwinsum;

  int tempn = nNote * _frameSize;

  Real *tempkernel;

  tempkernel = new Real[tempn];

  logFreqMatrix(_sampleRate, _frameSize, tempkernel);
  _kernelValue.clear();
  _kernelFftIndex.clear();
  _kernelNoteIndex.clear();
  int countNonzero = 0;
  for (int iNote = 0; iNote < nNote; ++iNote) { // I don't know if this is wise: manually making a sparse matrix
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

  _dict = new Real[nNote * 84];
  for (int i = 0; i < nNote * 84; ++i) _dict[i] = 0.0;

  dictionaryMatrix(_dict, _spectralShape);
}


void NNLS::compute() {
  const vector<vector<Real> >& logSpectrum = _logSpectrum.get();
  const vector<Real>& meanTuning = _meanTuning.get();
  const vector<Real>& localTuning = _localTuning.get();
  vector<vector<Real> >& tunedLogfreqSpectrum = _tunedLogfreqSpectrum.get();
  vector<vector<Real> >& semitoneSpectrum = _semitoneSpectrum.get();
  vector<vector<Real> >& bassChromagram = _bassChromagram.get();
  vector<vector<Real> >& chromagram = _chromagram.get();


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
        
    // interpolate all inner bins
    for (int k = 2; k < (int)logSpectrum[i].size() - 3; ++k) {
        tempValue = logSpectrum[i][k + intShift] * (1-RealShift) + logSpectrum[i][k+intShift+1] * RealShift;
        tunedLogfreqSpectrum[i].push_back(tempValue);
    }
        
    tunedLogfreqSpectrum[i].push_back(0.0); tunedLogfreqSpectrum[i].push_back(0.0); tunedLogfreqSpectrum[i].push_back(0.0); // upper edge

    vector<Real> runningmean = SpecialConvolution(tunedLogfreqSpectrum[i], _hw);
    vector<Real> runningstd;

    // first step: squared values into vector (variance)
    for (int j = 0; j < nNote; j++) {
      runningstd.push_back((tunedLogfreqSpectrum[i][j] - runningmean[j]) * (tunedLogfreqSpectrum[i][j] - runningmean[j]));
    }

    // second step convolve
    runningstd = SpecialConvolution(runningstd, _hw);

    for (int j = 0; j < nNote; j++) {  
      runningstd[j] = sqrt(runningstd[j]); // square root to finally have running std
      if (runningstd[j] > 0) {
        tunedLogfreqSpectrum[i][j] = (tunedLogfreqSpectrum[i][j] - runningmean[j]) > 0 ?
          (tunedLogfreqSpectrum[i][j] - runningmean[j]) / pow(runningstd[j], _whitening) : 0;
      }
      if (tunedLogfreqSpectrum[i][j] < 0) {
        E_INFO("ERROR: negative value in logfreq spectrum");
      }
    }
  }


  /** Semitone spectrum and chromagrams
      Semitone-spaced log-frequency spectrum derived from the tuned log-freq spectrum above. the spectrum
      is inferred using a non-negative least squares algorithm.
      Three different kinds of chromagram are calculated, "treble", "bass", and "both" (which means 
      bass and treble stacked onto each other).
  **/

  semitoneSpectrum.assign(logSpectrum.size(), vector<Real>());
  chromagram.assign(logSpectrum.size(), vector<Real>());
  bassChromagram.assign(logSpectrum.size(), vector<Real>());

  for (int i = 0; i < (int)logSpectrum.size(); i++) {
    // vector<Real> b(nNote, 0.f);
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

    // here's where the non-negative least squares algorithm calculates the note activation x
    vector<Real> chroma = vector<Real>(12, 0);
    vector<Real> basschroma = vector<Real>(12, 0);
    Real currval;
    int iSemitone = 0;

    if (some_b_greater_zero) {
      if (!_useNNLS) {
        for (int iNote = nBPS/2 + 2; iNote < nNote - nBPS/2; iNote += nBPS) {
          currval = 0;
          for (int iBPS = -nBPS/2; iBPS < nBPS/2+1; ++iBPS) {
            currval += b[iNote + iBPS] * (1-abs(iBPS*1.0/(nBPS/2+1)));						
          }

          semitoneSpectrum[i].push_back(currval);
          chroma[iSemitone % 12] += currval * treblewindow[iSemitone];
          basschroma[iSemitone % 12] += currval * basswindow[iSemitone];
          iSemitone++;
        } 
      }

      else {
        Real x[84+1000];
        for (int j = 1; j < 1084; ++j) x[j] = 1.0;

        vector<int> signifIndex;
        int index = 0;
        sumb /= 84.0;

        for (int iNote = nBPS/2 + 2; iNote < nNote - nBPS/2; iNote += nBPS) {
          Real currval = 0.f;
          for (int iBPS = -nBPS/2; iBPS < nBPS/2+1; ++iBPS) {
            currval += b[iNote + iBPS]; 
          }
          if (currval > 0.f) signifIndex.push_back(index);
          semitoneSpectrum[i].push_back(0.f); // fill the values, change later
          index++;
        }
        Real rnorm;
        Real w[84+1000];
        Real zz[84+1000];
        int indx[84+1000];
        int mode;
        int dictsize = nNote * signifIndex.size();
        Real *curr_dict = new Real[dictsize];
        for (int iNote = 0; iNote < (int)signifIndex.size(); ++iNote) {
          for (int iBin = 0; iBin < nNote; iBin++) {
            curr_dict[iNote * nNote + iBin] = 1.0 * _dict[signifIndex[iNote] * nNote + iBin];
          }
        }

        nnls(curr_dict, nNote, nNote, signifIndex.size(), b, x, &rnorm, w, zz, indx, &mode);
        delete [] curr_dict;

        for (int iNote = 0; iNote < (int)signifIndex.size(); ++iNote) {
          semitoneSpectrum[i][signifIndex[iNote]] = x[iNote];
          chroma[signifIndex[iNote] % 12] += x[iNote] * treblewindow[signifIndex[iNote]];
          basschroma[signifIndex[iNote] % 12] += x[iNote] * basswindow[signifIndex[iNote]];
        }
      }
    } 
    
    else {
      for (int j = 0; j < 84; ++j) semitoneSpectrum[i].push_back(0);
    }

    chromagram[i] = chroma;
    bassChromagram[i] = basschroma;

    if (_doNormalizeChroma > 0) {
      vector<Real> chromanorm = vector<Real>(3,0);			
      
      switch (_doNormalizeChroma) {
      case 0: // should never end up here
        break;
      case 1:
        chromanorm[0] = *max_element(chromagram[i].begin(), chromagram[i].end());
        chromanorm[1] = *max_element(bassChromagram[i].begin(), bassChromagram[i].end());
        chromanorm[2] = max(chromanorm[0], chromanorm[1]);
        break;
      case 2:
        for (vector<Real>::iterator it = chromagram[i].begin(); it != chromagram[i].end(); ++it) {
          chromanorm[0] += *it; 						
        }
        for (vector<Real>::iterator it = bassChromagram[i].begin(); it != bassChromagram[i].end(); ++it) {
          chromanorm[1] += *it; 						
        }
        break;
      case 3:
        for (vector<Real>::iterator it = chromagram[i].begin(); it != chromagram[i].end(); ++it) {
          chromanorm[0] += pow(*it,2); 						
        }
        chromanorm[0] = sqrt(chromanorm[0]);
        for (vector<Real>::iterator it = bassChromagram[i].begin(); it != bassChromagram[i].end(); ++it) {
          chromanorm[1] += pow(*it,2); 						
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

vector<Real> NNLS::SpecialConvolution(vector<Real> convolvee, vector<Real> kernel) {
    Real s;
    int m, n;
    int lenConvolvee = convolvee.size();
    int lenKernel = kernel.size();

    vector<Real> Z(nNote,0);
    assert(lenKernel % 2 != 0); // no exception handling !!!
    
    for (n = lenKernel - 1; n < lenConvolvee; n++) {
    	s=0.0;
    	for (m = 0; m < lenKernel; m++) {
            s += convolvee[n-m] * kernel[m];
    	}
        Z[n -lenKernel/2] = s;
    }
    
    // fill upper and lower pads
    for (n = 0; n < lenKernel/2; n++) Z[n] = Z[lenKernel/2];    
    for (n = lenConvolvee; n < lenConvolvee +lenKernel/2; n++) Z[n - lenKernel/2] = 
                                                                   Z[lenConvolvee - lenKernel/2 -  1];
    return Z;
}


#define nnls_max(a,b) ((a) >= (b) ? (a) : (b))
#define nnls_abs(x) ((x) >= 0 ? (x) : -(x))

/* Table of constant values */

int c__1 = 1;
int c__0 = 0;
int c__2 = 2;


Real NNLS::d_sign(Real a, Real b)
{
  Real x;
  x = (a >= 0 ? a : - a);
  return (b >= 0 ? x : -x);
}


int NNLS::g1(Real* a, Real* b, Real* cterm, Real* sterm, Real* sig)
{
  /* System generated locals */
  Real d;

  Real xr, yr;


  if (nnls_abs(*a) > nnls_abs(*b)) {
    xr = *b / *a;
    /* Computing 2nd power */
    d = xr;
    yr = sqrt(d * d + 1.);
    d = 1. / yr;
    *cterm = d_sign(d, *a);
    *sterm = *cterm * xr;
    *sig = nnls_abs(*a) * yr;
    return 0;
  }
  if (*b != 0.) {
    xr = *a / *b;
    /* Computing 2nd power */
    d = xr;
    yr = sqrt(d * d + 1.);
    d = 1. / yr;
    *sterm = d_sign(d, *b);
    *cterm = *sterm * xr;
    *sig = nnls_abs(*b) * yr;
    return 0;
  }
  *sig = 0.;
  *cterm = 0.;
  *sterm = 1.;
  return 0;
} /* g1_ */


int NNLS::nnls(Real* a,  int mda,  int m,  int n, Real* b,
               Real* x, Real* rnorm, Real* w, Real* zz, int* index,
               int* mode)
{
  /* System generated locals */
  int a_dim1, a_offset, idx1, idx2;
  Real d1, d2;


  /* Local variables */
  int iter;
  Real temp, wmax;
  int i__, j, l;
  Real t, alpha, asave;
  int itmax, izmax, nsetp;
  Real unorm, ztest, cc;
  Real dummy[2];
  int ii, jj, ip;
  Real sm;
  int iz, jz;
  Real up, ss;
  int rtnkey, iz1, iz2, npp1;

  /*     ------------------------------------------------------------------
   */
  /*     int INDEX(N) */
  /*     Real precision A(MDA,N), B(M), W(N), X(N), ZZ(M) */
  /*     ------------------------------------------------------------------
   */
  /* Parameter adjustments */
  a_dim1 = mda;
  a_offset = a_dim1 + 1;
  a -= a_offset;
  --b;
  --x;
  --w;
  --zz;
  --index;

  /* Function Body */
  *mode = 1;
  if (m <= 0 || n <= 0) {
    *mode = 2;
    return 0;
  }
  iter = 0;
  itmax = n * 3;

  /*                    INITIALIZE THE ARRAYS INDEX() AND X(). */

  idx1 = n;
  for (i__ = 1; i__ <= idx1; ++i__) {
    x[i__] = 0.;
    /* L20: */
    index[i__] = i__;
  }

  iz2 = n;
  iz1 = 1;
  nsetp = 0;
  npp1 = 1;
  /*                             ******  MAIN LOOP BEGINS HERE  ****** */
 L30:
  /*                  QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
   */
  /*                        OR IF M COLS OF A HAVE BEEN TRIANGULARIZED. */

  if (iz1 > iz2 || nsetp >= m) {
    goto L350;
  }

  /*         COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W().
   */

  idx1 = iz2;
  for (iz = iz1; iz <= idx1; ++iz) {
    j = index[iz];
    sm = 0.;
    idx2 = m;
    for (l = npp1; l <= idx2; ++l) {
      /* L40: */
      sm += a[l + j * a_dim1] * b[l];
    }
    w[j] = sm;
    /* L50: */
  }
  /*                                   FIND LARGEST POSITIVE W(J). */
 L60:
  wmax = 0.;
  idx1 = iz2;
  for (iz = iz1; iz <= idx1; ++iz) {
    j = index[iz];
    if (w[j] > wmax) {
      wmax = w[j];
      izmax = iz;
    }
    /* L70: */
  }

  /*             IF WMAX .LE. 0. GO TO TERMINATION. */
  /*             THIS INDICATES SATISFACTION OF THE KUHN-TUCKER CONDITIONS.
   */

  if (wmax <= 0.) {
    goto L350;
  }
  iz = izmax;
  j = index[iz];

  /*     THE SIGN OF W(J) IS OK FOR J TO BE MOVED TO SET P. */
  /*     BEGIN THE TRANSFORMATION AND CHECK NEW DIAGONAL ELEMENT TO AVOID */
  /*     NEAR LINEAR DEPENDENCE. */

  asave = a[npp1 + j * a_dim1];
  idx1 = npp1 + 1;
  h12(c__1, &npp1, &idx1, m, &a[j * a_dim1 + 1], &c__1, &up, dummy, &
      c__1, &c__1, &c__0);
  unorm = 0.;
  if (nsetp != 0) {
    idx1 = nsetp;
    for (l = 1; l <= idx1; ++l) {
      /* L90: */
      /* Computing 2nd power */
      d1 = a[l + j * a_dim1];
      unorm += d1 * d1;
    }
  }
  unorm = sqrt(unorm);
  d2 = unorm + (d1 = a[npp1 + j * a_dim1], nnls_abs(d1)) * .01;
  if ((d2- unorm) > 0.) {

    /*        COL J IS SUFFICIENTLY INDEPENDENT.  COPY B INTO ZZ, UPDATE Z
              Z */
    /*        AND SOLVE FOR ZTEST ( = PROPOSED NEW VALUE FOR X(J) ). */

    idx1 = m;
    for (l = 1; l <= idx1; ++l) {
      /* L120: */
      zz[l] = b[l];
    }
    idx1 = npp1 + 1;
    h12(c__2, &npp1, &idx1, m, &a[j * a_dim1 + 1], &c__1, &up, (zz+1), &
        c__1, &c__1, &c__1);
    ztest = zz[npp1] / a[npp1 + j * a_dim1];

    /*                                     SEE IF ZTEST IS POSITIVE */

    if (ztest > 0.) {
      goto L140;
    }
  }

  /*     REJECT J AS A CANDIDATE TO BE MOVED FROM SET Z TO SET P. */
  /*     RESTORE A(NPP1,J), SET W(J)=0., AND LOOP BACK TO TEST DUAL */
  /*     COEFFS AGAIN. */

  a[npp1 + j * a_dim1] = asave;
  w[j] = 0.;
  goto L60;

  /*     THE INDEX  J=INDEX(IZ)  HAS BEEN SELECTED TO BE MOVED FROM */
  /*     SET Z TO SET P.    UPDATE B,  UPDATE INDICES,  APPLY HOUSEHOLDER */
  /*     TRANSFORMATIONS TO COLS IN NEW SET Z,  ZERO SUBDIAGONAL ELTS IN */
  /*     COL J,  SET W(J)=0. */

 L140:
  idx1 = m;
  for (l = 1; l <= idx1; ++l) {
    /* L150: */
    b[l] = zz[l];
  }

  index[iz] = index[iz1];
  index[iz1] = j;
  ++iz1;
  nsetp = npp1;
  ++npp1;

  if (iz1 <= iz2) {
    idx1 = iz2;
    for (jz = iz1; jz <= idx1; ++jz) {
      jj = index[jz];
      h12(c__2, &nsetp, &npp1, m,
          &a[j * a_dim1 + 1], &c__1, &up,
          &a[jj * a_dim1 + 1], &c__1, &mda, &c__1);
      /* L160: */
    }
  }

  if (nsetp != m) {
    idx1 = m;
    for (l = npp1; l <= idx1; ++l) {
      /* L180: */
      // SS: CHECK THIS DAMAGE....
      a[l + j * a_dim1] = 0.;
    }
  }

  w[j] = 0.;
  /*                                SOLVE THE TRIANGULAR SYSTEM. */
  /*                                STORE THE SOLUTION TEMPORARILY IN ZZ().
   */
  rtnkey = 1;
  goto L400;
 L200:

  /*                       ******  SECONDARY LOOP BEGINS HERE ****** */

  /*                          ITERATION COUNTER. */

 L210:
  ++iter;
  if (iter > itmax) {
    *mode = 3;
    goto L350;
  }

  /*                    SEE IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE. */
  /*                                  IF NOT COMPUTE ALPHA. */

  alpha = 2.;
  idx1 = nsetp;
  for (ip = 1; ip <= idx1; ++ip) {
    l = index[ip];
    if (zz[ip] <= 0.) {
      t = -x[l] / (zz[ip] - x[l]);
      if (alpha > t) {
        alpha = t;
        jj = ip;
      }
    }
    /* L240: */
  }

  /*          IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE THEN ALPHA WILL */
  /*          STILL = 2.    IF SO EXIT FROM SECONDARY LOOP TO MAIN LOOP. */

  if (alpha == 2.) {
    goto L330;
  }

  /*          OTHERWISE USE ALPHA WHICH WILL BE BETWEEN 0. AND 1. TO */
  /*          INTERPOLATE BETWEEN THE OLD X AND THE NEW ZZ. */

  idx1 = nsetp;
  for (ip = 1; ip <= idx1; ++ip) {
    l = index[ip];
    x[l] += alpha * (zz[ip] - x[l]);
    /* L250: */
  }

  /*        MODIFY A AND B AND THE INDEX ARRAYS TO MOVE COEFFICIENT I */
  /*        FROM SET P TO SET Z. */

  i__ = index[jj];
 L260:
  x[i__] = 0.;

  if (jj != nsetp) {
    ++jj;
    idx1 = nsetp;
    for (j = jj; j <= idx1; ++j) {
      ii = index[j];
      index[j - 1] = ii;
      g1(&a[j - 1 + ii * a_dim1], &a[j + ii * a_dim1],
         &cc, &ss, &a[j - 1 + ii * a_dim1]);
      // SS: CHECK THIS DAMAGE...
      a[j + ii * a_dim1] = 0.;
      idx2 = n;
      for (l = 1; l <= idx2; ++l) {
        if (l != ii) {

          /*                 Apply procedure G2 (CC,SS,A(J-1,L),A(J,
                             L)) */

          temp = a[j - 1 + l * a_dim1];
          // SS: CHECK THIS DAMAGE
          a[j - 1 + l * a_dim1] = cc * temp + ss * a[j + l * a_dim1];
          a[j + l * a_dim1] = -ss * temp + cc * a[j + l * a_dim1];
        }
        /* L270: */
      }

      /*                 Apply procedure G2 (CC,SS,B(J-1),B(J)) */

      temp = b[j - 1];
      b[j - 1] = cc * temp + ss * b[j];
      b[j] = -ss * temp + cc * b[j];
      /* L280: */
    }
  }

  npp1 = nsetp;
  --nsetp;
  --iz1;
  index[iz1] = i__;

  /*        SEE IF THE REMAINING COEFFS IN SET P ARE FEASIBLE.  THEY SHOULD
   */
  /*        BE BECAUSE OF THE WAY ALPHA WAS DETERMINED. */
  /*        IF ANY ARE INFEASIBLE IT IS DUE TO ROUND-OFF ERROR.  ANY */
  /*        THAT ARE NONPOSITIVE WILL BE SET TO ZERO */
  /*        AND MOVED FROM SET P TO SET Z. */

  idx1 = nsetp;
  for (jj = 1; jj <= idx1; ++jj) {
    i__ = index[jj];
    if (x[i__] <= 0.) {
      goto L260;
    }
    /* L300: */
  }

  /*         COPY B( ) INTO ZZ( ).  THEN SOLVE AGAIN AND LOOP BACK. */

  idx1 = m;
  for (i__ = 1; i__ <= idx1; ++i__) {
    /* L310: */
    zz[i__] = b[i__];
  }
  rtnkey = 2;
  goto L400;
 L320:
  goto L210;
  /*                      ******  END OF SECONDARY LOOP  ****** */

 L330:
  idx1 = nsetp;
  for (ip = 1; ip <= idx1; ++ip) {
    i__ = index[ip];
    /* L340: */
    x[i__] = zz[ip];
  }
  /*        ALL NEW COEFFS ARE POSITIVE.  LOOP BACK TO BEGINNING. */
  goto L30;

  /*                        ******  END OF MAIN LOOP  ****** */

  /*                        COME TO HERE FOR TERMINATION. */
  /*                     COMPUTE THE NORM OF THE FINAL RESIDUAL VECTOR. */

 L350:
  sm = 0.;
  if (npp1 <= m) {
    idx1 = m;
    for (i__ = npp1; i__ <= idx1; ++i__) {
      /* L360: */
      /* Computing 2nd power */
      d1 = b[i__];
      sm += d1 * d1;
    }
  } else {
    idx1 = n;
    for (j = 1; j <= idx1; ++j) {
      /* L380: */
      w[j] = 0.;
    }
  }
  *rnorm = sqrt(sm);
  return 0;

  /*     THE FOLLOWING BLOCK OF CODE IS USED AS AN INTERNAL SUBROUTINE */
  /*     TO SOLVE THE TRIANGULAR SYSTEM, PUTTING THE SOLUTION IN ZZ(). */

 L400:
  idx1 = nsetp;
  for (l = 1; l <= idx1; ++l) {
    ip = nsetp + 1 - l;
    if (l != 1) {
      idx2 = ip;
      for (ii = 1; ii <= idx2; ++ii) {
        zz[ii] -= a[ii + jj * a_dim1] * zz[ip + 1];
        /* L410: */
      }
    }
    jj = index[ip];
    zz[ip] /= a[ip + jj * a_dim1];
    /* L430: */
  }
  switch ((int)rtnkey) {
  case 1:  goto L200;
  case 2:  goto L320;
  }

  return 0;

} /* nnls_ */


int NNLS::h12(int mode, int* lpivot, int* l1,
      int m, Real* u, int* iue, Real* up, Real* c__,
        int* ice, int* icv, int* ncv)
{
  /* System generated locals */
  int u_dim1, u_offset, idx1, idx2;
Real d, d2;

  /* Local variables */
  int incr;
Real b;
  int i__, j;
Real clinv;
  int i2, i3, i4;
Real cl, sm;

  /*     ------------------------------------------------------------------
   */
/*     Real precision U(IUE,M) */
  /*     ------------------------------------------------------------------
   */
  /* Parameter adjustments */
  u_dim1 = *iue;
  u_offset = u_dim1 + 1;
  u -= u_offset;
  --c__;

  /* Function Body */
  if (0 >= *lpivot || *lpivot >= *l1 || *l1 > m) {
    return 0;
  }
  cl = (d = u[*lpivot * u_dim1 + 1], nnls_abs(d));
  if (mode == 2) {
    goto L60;
  }
  /*                            ****** CONSTRUCT THE TRANSFORMATION. ******
   */
  idx1 = m;
  for (j = *l1; j <= idx1; ++j) {
    /* L10: */
    /* Computing MAX */
    d2 = (d = u[j * u_dim1 + 1], nnls_abs(d));
    cl = nnls_max(d2,cl);
  }
  if (cl <= 0.) {
    goto L130;
  } else {
    goto L20;
  }
 L20:
  clinv = 1. / cl;
  /* Computing 2nd power */
  d = u[*lpivot * u_dim1 + 1] * clinv;
  sm = d * d;
  idx1 = m;
  for (j = *l1; j <= idx1; ++j) {
    /* L30: */
    /* Computing 2nd power */
    d = u[j * u_dim1 + 1] * clinv;
    sm += d * d;
  }
  cl *= sqrt(sm);
  if (u[*lpivot * u_dim1 + 1] <= 0.) {
    goto L50;
  } else {
    goto L40;
  }
 L40:
  cl = -cl;
 L50:
  *up = u[*lpivot * u_dim1 + 1] - cl;
  u[*lpivot * u_dim1 + 1] = cl;
  goto L70;
  /*            ****** APPLY THE TRANSFORMATION  I+U*(U**T)/B  TO C. ******
   */

 L60:
  if (cl <= 0.) {
    goto L130;
  } else {
    goto L70;
  }
 L70:
  if (*ncv <= 0) {
    return 0;
  }
  b = *up * u[*lpivot * u_dim1 + 1];
  /*                       B  MUST BE NONPOSITIVE HERE.  IF B = 0., RETURN.
   */

  if (b >= 0.) {
    goto L130;
  } else {
    goto L80;
  }
 L80:
  b = 1. / b;
  i2 = 1 - *icv + *ice * (*lpivot - 1);
  incr = *ice * (*l1 - *lpivot);
  idx1 = *ncv;
  for (j = 1; j <= idx1; ++j) {
    i2 += *icv;
    i3 = i2 + incr;
    i4 = i3;
    sm = c__[i2] * *up;
    idx2 = m;
    for (i__ = *l1; i__ <= idx2; ++i__) {
      sm += c__[i3] * u[i__ * u_dim1 + 1];
      /* L90: */
      i3 += *ice;
    }
    if (sm != 0.) {
      goto L100;
    } else {
      goto L120;
    }
  L100:
    sm *= b;
    c__[i2] += sm * *up;
    idx2 = m;
    for (i__ = *l1; i__ <= idx2; ++i__) {
      c__[i4] += sm * u[i__ * u_dim1 + 1];
      /* L110: */
      i4 += *ice;
    }
  L120:
    ;
  }
 L130:
  return 0;
} /* h12 */


/**
 * Calculates a matrix that can be used to linearly map from the magnitude spectrum to a pitch-scale spectrum.
 * @return this always returns true, which is a bit stupid, really. The main purpose of the function is to change the values in the "matrix" pointed to by *outmatrix
 */
bool NNLS::logFreqMatrix(Real fs, int frameSize, Real *outmatrix) {
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
    // cerr << cosp << endl;
  }

  for (int i = 0; i < nFFT * (int)cq_f.size(); ++i) {
    outmatrix[i] = 0.f;
  }

  Real cq_activation;
  for (int iFFT = 1; iFFT < nFFT; ++iFFT) {
    // find frequency stretch where the oversampled vector can be non-zero (i.e. in a window of width fft_width around the current frequency)
    int curr_start = oversampling * iFFT - oversampling;
    int curr_end = oversampling * iFFT + oversampling; // don't know if I should add "+1" here
    // cerr << oversampled_f[curr_start] << " " << fft_f[iFFT] << " " << oversampled_f[curr_end] << endl;
    for (int iCQ = 0; iCQ < (int)cq_f.size(); ++iCQ) {
      if (cq_f[iCQ] * pow(2.0, 0.084) + fft_width > fft_f[iFFT] && cq_f[iCQ] * pow(2.0, -0.084 * 2) - fft_width < fft_f[iFFT]) { // within a generous neighbourhood
        for (int iOS = curr_start; iOS < curr_end; ++iOS) {
          cq_activation = pitchCospuls(oversampled_f[iOS],cq_f[iCQ],binspersemitone*12);
          // cerr << oversampled_f[iOS] << " " << cq_f[iCQ] << " " << cq_activation << endl;
          outmatrix[iFFT + nFFT * iCQ] += cq_activation * fft_activation[iOS-curr_start];
        }
      }
    }
  }
  return true;	
}


Real NNLS::cospuls(Real x, Real centre, Real width) {
  Real recipwidth = 1.0/width;
  if (abs(x - centre) <= 0.5 * width) {
    return cos((x-centre)*2*M_PI*recipwidth)*.5+.5;
  }
  return 0.0;
}

Real NNLS::pitchCospuls(Real x, Real centre, int binsperoctave) {
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

void NNLS::dictionaryMatrix(Real* dm, Real s_param) {
	// TODO: make this more general, such that it works with all minoctave, maxoctave and even more than one note per semitone
    int binspersemitone = nBPS;
    int minoctave = 0; // this must be 0
    int maxoctave = 7; // this must be 7
	
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
    cq_f.push_back(440 * pow(2.0,0.083333 * (maxMIDI-69)));

//    Real curr_f;
    Real Realbin;
    Real curr_amp;
    // now for every combination calculate the matrix element
    for (int iOut = 0; iOut < 12 * (maxoctave - minoctave); ++iOut) {
        // cerr << iOut << endl;
        for (int iHarm = 1; iHarm <= 20; ++iHarm) {
//            curr_f = 440 * pow(2,(minMIDI-69+iOut)*1.0/12) * iHarm;
            // if (curr_f > cq_f[nNote-1])  break;
            Realbin = ((iOut + 1) * binspersemitone + 1) + binspersemitone * 12 * log2(iHarm);
            // cerr << Realbin << endl;
            curr_amp = pow(s_param,Real(iHarm-1));
            // cerr << "curramp" << curr_amp << endl;
            for (int iNote = 0; iNote < nNote; ++iNote) {
                if (abs(iNote+1.0-Realbin)<2) {
                    dm[iNote  + nNote * iOut] += cospuls(iNote+1.0, Realbin, binspersemitone + 0.0) * curr_amp;
                    // dm[iNote + nNote * iOut] += 1 * curr_amp;
                }
            }
        }
    }
}