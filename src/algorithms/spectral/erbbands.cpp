/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 */

#include "erbbands.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* ERBBands::name = "ERBBands";
const char* ERBBands::version = "1.0";
const char* ERBBands::description = DOC("This algorithm computes energies/magnitudes in bands spaced on an Equivalent Rectangular Bandwidth (ERB) scale, given a spectrum. It applies a frequency domain filterbank using gammatone filters. Adapted from matlab code in:  D. P. W. Ellis (2009). 'Gammatone-like spectrograms', web resource [1].\n"
"\n"
"References:\n"
"  [1] http://www.ee.columbia.edu/~dpwe/resources/matlab/gammatonegram/\n\n"
"  [2] B. C. Moore and B. R. Glasberg, \"Suggested formulae for calculating\n"
"  auditory-filter bandwidths and excitation patterns,\" Journal of the\n"
"  Acoustical Society of America, vol. 74, no. 3, pp. 750–753, 1983.");

const Real ERBBands::EarQ = 9.26449;
const Real ERBBands::minBW = 24.7;


void ERBBands::configure() {
  if (parameter("highFrequencyBound").toReal() >
        parameter("sampleRate").toReal()*0.5 ) {
    throw EssentiaException(
        "ERBBands: High frequency bound cannot be higher than Nyquist frequency");
  }
  if (parameter("highFrequencyBound").toReal() <=
        parameter("lowFrequencyBound").toReal()) {
    throw EssentiaException(
        "ERBands: High frequency bound cannot be lower than low frequency bound");
  }

  _numberBands = parameter("numberBands").toInt();
  _sampleRate = parameter("sampleRate").toReal();
  _maxFrequency = parameter("highFrequencyBound").toReal();
  _minFrequency = parameter("lowFrequencyBound").toReal();
  _width = parameter("width").toReal();
  calculateFilterFrequencies();
  createFilters(parameter("inputSize").toInt());

  _type = parameter("type").toLower();
}

void ERBBands::calculateFilterFrequencies() {
  int filterSize = _numberBands;
  _filterFrequencies.resize(filterSize);
  Real filterSizeInv = 1.0/filterSize;
  Real bw = EarQ*minBW;

  for (int i=1; i<filterSize+1; i++) {
	_filterFrequencies[filterSize-i] = -bw +
		exp(i*(-log(_maxFrequency + bw) + log(_minFrequency + bw)) * filterSizeInv) * (_maxFrequency + bw);
  }
}

void ERBBands::createFilters(int spectrumSize) {
  if (spectrumSize < 2) {
    throw EssentiaException("ERBBands: Filter bank cannot be computed from a spectrum with less than 2 bins");
  }

  int filterSize = _numberBands;
  vector<complex<Real> > ucirc = vector<complex<Real> >(spectrumSize);
  complex<Real> oneJ(0,1);
  Real order = 1;
  Real pi = Real(M_PI);
  _filterCoefficients = vector<vector<Real> >(filterSize, vector<Real>(spectrumSize, 0.0));
  Real fftSize = (spectrumSize-1)*2;
  for (int i=0; i<spectrumSize; i++) {
 	  ucirc[i] = exp((oneJ*Real(2.0)*pi*Real(i))/fftSize);
  }

  Real sqrP = sqrt(3+pow(2,1.5));
  Real sqrM = sqrt(3-pow(2,1.5));


  for (int i=0; i<filterSize; i++) {
    Real cf = _filterFrequencies[i];
    Real ERB = _width*pow((pow((cf/EarQ),order) +
                          pow(minBW,order)),Real(1.0/order));
    Real B = Real(1.019)*2*M_PI*ERB;
    Real r = exp(-B/ _sampleRate);
    Real theta = Real(2)*M_PI*cf/ _sampleRate;
    complex<Real> pole = r*exp(oneJ*theta);
    Real T = 1.0/ _sampleRate;
    Real GTord = 4;

    Real sinCf =  sin(2*cf*pi*T);
    Real cosCf = cos(2*cf*pi*T);
    Real gtCos = 2*T*cosCf/exp(B*T);
    Real gtSin = T*sinCf/exp(B*T);

    Real A11 = -(gtCos + 2*sqrP * gtSin)/2;
    Real A12 = -(gtCos - 2*sqrP * gtSin)/2;
    Real A13 = -(gtCos + 2*sqrM * gtSin)/2;
    Real A14 = -(gtCos - 2*sqrM * gtSin)/2;

	  vector<Real> zeros = vector<Real>(4);
	  zeros[0] = - A11 / T;
	  zeros[1] = - A12 / T;
	  zeros[2] = - A13 / T;
	  zeros[3] = - A14 / T;

    complex<Real> g1 = Real(-2)*exp(Real(4)*oneJ*cf*pi*T)*T;
    complex<Real> g2 = Real(2)*exp(-(B*T) + Real(2)*oneJ*cf*pi*T)*T;
    complex<Real> cxExp = exp(Real(4)*oneJ*cf*pi*T);

    Real filterGain = abs(
      (g1 + g2 *(cosCf - sqrM *sinCf)) *
      (g1 + g2 *(cosCf + sqrM *sinCf)) *
      (g1 +g2 * (cosCf - sqrP *sinCf)) *
      (g1 +g2 * (cosCf + sqrP *sinCf)) /
      pow((Real(-2) / exp(Real(2)*B*T) -
                Real(2)* cxExp + Real(2)*(Real(1) + cxExp)/exp(B*T)),Real(4)));

    for (int j=0; j<spectrumSize; j++) {
      _filterCoefficients[i][j] = (pow(T,4)/filterGain) *
            abs(ucirc[j]-zeros[0]) * abs(ucirc[j]-zeros[1]) *
            abs(ucirc[j]-zeros[2]) * abs(ucirc[j]-zeros[3]) *
            pow(abs((pole-ucirc[j])*(pole-ucirc[j])),(-GTord));
    }
  }
}

void ERBBands::compute() {

  const std::vector<Real>& spectrum = _spectrumInput.get();
  std::vector<Real>& bands = _bandsOutput.get();

  int filterSize = _numberBands;
  int spectrumSize = spectrum.size();

  if (_filterCoefficients.empty() ||
      int(_filterCoefficients[0].size()) != spectrumSize) {
    cout << "ERBBands: input spectrum size does not correspond to the \"inputSize\" parameter. Recomputing the filter bank." << endl;
    createFilters(spectrumSize);
  }

  bands.resize(filterSize);


  // NB: Band magnitudes are returned, while BarkBands and MelBands algorithms
  // return energy. Gerard Roma have found magnitudes work better when
  // working with sound effects.  Band magnitudes option is required for 
  // OnsetDetectionGlobal algorithm.

  // TODO: probably all *Bands algorithms should have an option {magnitude,energy} 

  if (_type=="magnitude") {
    for (int i=0; i<filterSize; ++i) {
      bands[i] = 0;
      for (int j=0; j<spectrumSize; ++j) {
        bands[i] += (spectrum[j]) * _filterCoefficients[i][j];
      }
    }
  }
  else if (_type=="energy") {
    for (int i=0; i<filterSize; ++i) {
      bands[i] = 0;
      for (int j=0; j<spectrumSize; ++j) {
        bands[i] += (spectrum[j] * spectrum[j]) * _filterCoefficients[i][j];
      }
    }
  }
}
