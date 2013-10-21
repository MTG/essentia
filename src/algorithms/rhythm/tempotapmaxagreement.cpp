/*
 * Copyright (C) 2006-2012 Music Technology Group (MTG)
 *                         Universitat Pompeu Fabra
 *
 */

#include "tempotapmaxagreement.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* TempoTapMaxAgreement::name = "TempoTapMaxAgreement";
const char* TempoTapMaxAgreement::description = DOC("This algorithm estimates beat positions and confidence of their estimation based on the maximum mutual agreement between given beat postion candidates, estimated by different beat trackers (or using different features) [1,2].\n"
"\n"
"Note that the input tick times should be in ascending order and that they cannot contain negative values otherwise an exception will be thrown.\n"
"\n"
"References:\n"
"  [1] J. R. Zapata, A. Holzapfel, M. E. Davies, J. L. Oliveira, and\n"
"  F. Gouyon, \"Assigning a confidence threshold on automatic beat annotation\n"
"  in large datasets,\" in International Society for Music Information\n" 
"  Retrieval Conference (ISMIR’12), 2012.\n\n"
"  [2] A. Holzapfel, M. E. Davies, J. R. Zapata, J. L. Oliveira, and\n"
"  F. Gouyon, \"Selective sampling for beat tracking evaluation,\" IEEE\n" 
"  Transactions on Audio, Speech, and Language Processing, vol. 13, no. 9,\n"
"  pp. 2539-2548, 2012.\n");


void TempoTapMaxAgreement::configure() {

  // assign histogram bin centers
  _histogramBins.reserve(_numberBins+1);
  _histogramBins.push_back(-0.5);
  Real delta = 1. / (_numberBins-1);
  for (Real bin=-0.5 + 1.5*delta; bin <= 0.5-1.5*delta; bin+=delta) {
    _histogramBins.push_back(bin);
  }
  _histogramBins.push_back(0.5);
  _binValues.resize(_histogramBins.size());

  // in practice, we will need bin borders instead of centers
  for (size_t i=0; i<_histogramBins.size()-1; ++i) {
    _histogramBins[i] = (_histogramBins[i] + _histogramBins[i+1]) / 2;
  }
  _histogramBins.pop_back();
}


void TempoTapMaxAgreement::reset() {
  Algorithm::reset();
}

void TempoTapMaxAgreement::compute() {
  vector<vector<Real> > tickCandidates = _tickCandidates.get(); // we need a copy
  vector<Real>& ticks = _ticks.get();
  Real& confidence = _confidence.get();

  // sanity checks
  for(int i=0; i<(int) tickCandidates.size(); ++i) {
    for (size_t j=0; j<tickCandidates[i].size(); ++j) {
      if (tickCandidates[i][j]<0) {
        throw EssentiaException("TempoTapMaxAgreement: tick values must be non-negative");
      }
      if (j>=1) {
        if (tickCandidates[i][j] <= tickCandidates[i][j-1]) {
          throw EssentiaException("TempoTapMaxAgreement: tick values must be in ascending order");
        }
      }
    }
  }

  ticks.clear();
  if (!tickCandidates.size()) {
    return; // no candidates were provided
  }

  // remove ticks that are within the first _minTickTime seconds
  for(int i=0; i<(int) tickCandidates.size(); ++i) {
    removeFirstSeconds(tickCandidates[i]);
  }

  int numberMethods = (int) tickCandidates.size();
  vector<vector<Real> > infogain(numberMethods, vector<Real> (numberMethods, 0.));

  for (int i=0; i<numberMethods; ++i) {
    for (int j=i+1; j<numberMethods; ++j) {
      infogain[i][j] = computeBeatInfogain(tickCandidates[i], tickCandidates[j]);
    }
  }

  vector<Real> temp1;
  temp1.reserve(2*numberMethods); // reserve more than maximum we will ever

  vector<Real> distanceInfogain;
  distanceInfogain.reserve(numberMethods);

  for (int i=0; i<numberMethods; ++i) {
    // gather all combinations in which i-th method was compared
    for (int j=i+1; j<numberMethods; ++j) {
      temp1.push_back(infogain[i][j]);
    }
    for (int j=0; j<i; ++j) {
      temp1.push_back(infogain[j][i]);
    }

    distanceInfogain.push_back(mean(temp1));
    temp1.clear();
  }

  int selectedMethod = argmax(distanceInfogain);
  ticks = _tickCandidates.get()[selectedMethod];
  confidence = mean(distanceInfogain);
}


Real TempoTapMaxAgreement::computeBeatInfogain(vector<Real>& ticks1,
                                               vector<Real>& ticks2) {

  // return zero information gain on empty or too short tick sequencies
  if (ticks1.size()<2 || ticks2.size()<2) {
    return 0;
  }

  vector<Real> forwardError;
  vector<Real> backwardError;

  // ticks2 compared to ticks1
  FindBeatError(ticks2, ticks1, forwardError);
  Real forwardEntropy = FindEntropy(forwardError);

  // ticks1 compared to ticks2
  FindBeatError(ticks1, ticks2, backwardError);
  Real backwardEntropy = FindEntropy(backwardError);

  // find higher entropy value (i.e. which is the worst)
  Real maxEntropy = max(forwardEntropy, backwardEntropy);
  return log2(_numberBins) - maxEntropy;
}


void TempoTapMaxAgreement::removeFirstSeconds(vector<Real>& ticks) {
  size_t removeTicks=0;
  for (; removeTicks<ticks.size(); ++removeTicks) {
    if (ticks[removeTicks] >= _minTickTime) break;
  }
  vector<Real>(ticks.begin()+removeTicks, ticks.end()).swap(ticks);
}


void TempoTapMaxAgreement::FindBeatError(const vector<Real>& ticks1,
                                         const vector<Real>& ticks2,
                                         vector<Real>& beatError) {
  beatError.reserve(ticks2.size());

  // Calculate relative error for each beat sample
  for (size_t i=0; i<ticks2.size(); ++i) {
    Real interval;

    // find the closest tick in tick1 to tick2[i]
    size_t j = closestTick(ticks1, ticks2[i]);
    Real error = ticks2[i] - ticks1[j];

    if (j==0) { // first tick is the nearest
      interval = 0.5*(ticks1[j+1] - ticks1[j]);
    }
    else if (j==ticks1.size()-1) {  // last tick is the nearest
      interval = 0.5*(ticks1[j] - ticks1[j-1]);
    }
    // test if the error is positive or negative and choose interval accordingly
    else if (error < 0) {
      // nearest tick is before ticks2[i] --> look at the previous interval
      interval = 0.5*(ticks1[j] - ticks1[j-1]);
    }
    else {
      // nearest tick is after ticks2[i] --> look at the next interval
      interval = 0.5*(ticks1[j+1] - ticks1[j]);
    }
    beatError.push_back(0.5 * error / interval); // relative error
  }

  // original matlab code: weird trick to deal with bin boundaries:
  // beatError = round(10000*beatError)/10000;
}


Real TempoTapMaxAgreement::FindEntropy(vector<Real>& beatError) {
  // fix the beat errors which are out of range in a way similar to princarg,
  // but for [-0.5, 0.5]

  for (size_t i=0; i<beatError.size(); ++i) {
    beatError[i] = fmod(beatError[i] + 0.5, 1.) - 0.5;
  }
  // compute the histogram
  histogram(beatError, _binValues);

  // add the last bin frequency to the first bin
  _binValues.front() += _binValues.back();
  _binValues.pop_back();  // remove and add back after the computations

  normalizeSum(_binValues);

  // compute the entropy
  Real entropy = 0.;
  for (size_t i=0; i<_binValues.size(); ++i) {
    if (!_binValues[i]) {  // set zero valued bins to 1
      _binValues[i] = 1;   // to make the entropy calculation well-behaved
    }
    entropy -= log2(_binValues[i]) * _binValues[i];
  }
  _binValues.push_back(0.);
  return entropy;
}


size_t TempoTapMaxAgreement::closestTick(const vector<Real>& ticks, Real x) {
  // find closest to x tick in ticks
  Real minDistance=-1;
  size_t j=0;

  while(j<ticks.size()) {
    Real distance = abs(ticks[j] - x);
    if (minDistance < 0) { // first comparision
      minDistance = distance;
    }
    else if (distance < minDistance) { // distances decrease
      minDistance = distance;
    }
    else break; // distances start increase, we have passed the minimum
    j++;
  }
  return j-1;
}


void TempoTapMaxAgreement::histogram(const vector<Real>& array, vector<Real>& counter) {
  counter.clear();
  counter.resize(_histogramBins.size()+1);
  for (size_t i=0; i<array.size(); ++i) {
    if (array[i] >= _histogramBins.back()) {
      counter.back() += 1;
    }
    else {
      for (size_t b=0; b<_histogramBins.size(); ++b) {
        if (array[i] < _histogramBins[b]) {
          counter[b] += 1;
          break;
        }
      }
    }
  }
}

