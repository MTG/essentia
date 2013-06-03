/*
 * Copyright (C) 2006-2008 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "meter.h"
#include "algorithmfactory.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* Meter::name = "Meter";
const char* Meter::version = "1.0";
const char* Meter::description = DOC("This algorithm estimates the time signature of a given beatogram by finding the highest correlation between beats.\n"
"This algorithm still experimental.\n");

void Meter::configure() {
}

bool Meter::isPowerN(int val, int power) {
  Real d = log(Real(val))/log(Real(power));
  return (d-int(d)) == 0;
}

bool Meter::isPowerHarmonic(int x, int y) {
  if (x<2 || y<2) return false;
  if (x<y) return isPowerHarmonic(y,x);
  return (x%y==0) && (isPowerTwo(x/y) || isPowerN(x,y));
}
void Meter::compute() {
  const vector<vector<Real> >& beatogram = _beatogram.get();
  Real& meter = _meter.get();
  int nbands= beatogram.size();
  if (nbands<1) {
    throw EssentiaException("Meter: empty beatogram");
  }
  int nticks = beatogram[0].size();

  AlgorithmFactory& factory = AlgorithmFactory::instance();
  Algorithm* acorr=factory.create("AutoCorrelation");
  vector<vector<Real> > bandCorr(nbands);
  Real maxBand=0, maxBandValue=0;
  vector<Real> maxCorr(nbands);
  vector<int> maxCorrIdx(nbands);
  for (int iBand=0; iBand<nbands; iBand++) {
     acorr->input("array").set(beatogram[iBand]);
     acorr->output("autoCorrelation").set(bandCorr[iBand]);
     acorr->compute();
     acorr->reset();
     const vector<Real>& corr = bandCorr[iBand];
     maxCorrIdx[iBand] = argmax(vector<Real>(corr.begin()+2, corr.end()))+2;
     maxCorr[iBand]=corr[maxCorrIdx[iBand]];
     if (maxCorr[iBand] > maxBandValue) {
       maxBandValue = maxCorr[iBand];
       maxBand = iBand;
     }
  }
  delete acorr;
  //cout << "bands max correlation: " << maxCorrIdx << endl;
  vector<Real> sumCorr(nticks, 0.0);
  for (int iTick=0; iTick<nticks; iTick++) {
    for (int iBand=0; iBand<nbands; iBand++) {
      sumCorr[iTick]+=bandCorr[iBand][iTick];
    }
  }
  Real maxSumCorr = argmax(vector<Real>(sumCorr.begin()+2, sumCorr.end()))+2;
  //cout << "bands max sum correlation: " << maxSumCorr << endl;
  meter = maxSumCorr;

  // compute harmonics on sumCorr:
  vector<Real> hist(nticks);
  vector<int> counts(nticks);
  for (int i=0; i<nticks; i++) {
    for (int j=0; j<nticks; j++) {
      if (isPowerHarmonic(i,j)) {
        hist[i] += sumCorr[j];
        counts[i]++;
      }
    }
  }
  for (int i=0; i<nticks; i++) {
    if (counts[i] > 0) hist[i] /= Real(counts[i]);
  }

  //cout << "hist: " << vector<Real>(hist.begin(),hist.begin()+16) << endl;

  // only up to the 16th bar:
  //cout << "bands max histogram: " << argmax(vector<Real>(hist.begin(),hist.begin()+16)) << endl;
}
