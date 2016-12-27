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

#include "sinemodelanal.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* SineModelAnal::name = "SineModelAnal";
const char* SineModelAnal::category = "Synthesis";
const char* SineModelAnal::description = DOC("This algorithm computes the sine model analysis. \n"
"\n"
"It is recommended that the input \"spectrum\" be computed by the Spectrum algorithm. This algorithm uses PeakDetection. See documentation for possible exceptions and input requirements on input \"spectrum\".\n"
"\n"
"References:\n"
"  [1] Peak Detection,\n"
"  http://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html");


// ------------------
// Additional support functions
//typedef std::pair<int,Real> mypair;
// sort indexes: get arguments of sorted vector
bool SineModelAnal::comparator_up ( const mypair& l, const mypair& r)
{
  return l.second < r.second;
}
bool SineModelAnal::comparator_down ( const mypair& l, const mypair& r)
{
  return l.second > r.second;
}

// It sorts the indexes of an input vector v, and outputs the sorted index vector idx
void SineModelAnal::sort_indexes(std::vector<int> &idx, const std::vector<Real> &v, bool ascending) {

  // initialize original index locations
  std::vector<mypair> pairs(v.size());
  for (int i = 0; i != (int)pairs.size(); ++i){
    pairs[i].first = i;
    pairs[i].second = v[i];
  }

  // sort indexes based on comparing values in v
  if (ascending)
    sort(pairs.begin(), pairs.end(),comparator_up);
  else
    sort(pairs.begin(), pairs.end(),comparator_down);

  // copy sorted indexes
  for (int i = 0; i != (int)pairs.size(); ++i) idx.push_back(pairs[i].first);

  return;
}

void SineModelAnal::copy_vector_from_indexes(std::vector<Real> &out, const std::vector<Real> v, const std::vector<int> idx){

  for (int i = 0; i < (int)idx.size(); ++i){
    out.push_back(v[idx[i]]);
  }
  return;
}

void SineModelAnal::copy_int_vector_from_indexes(std::vector<int> &out, const std::vector<int> v, const std::vector<int> idx){

  for (int i = 0; i < (int)idx.size(); ++i){
    out.push_back(v[idx[i]]);
  }
  return;
}

// erase elements from a vector given a vector of indexes
void SineModelAnal::erase_vector_from_indexes(std::vector<Real> &v, const std::vector<int> idx){
  std::vector<Real> tmp;
  bool found;
  for (int i = 0; i < (int)v.size(); ++i) {
    found = false;
    for (int j = 0; j < (int)idx.size(); ++j){
      if (i == idx[j])
        found = true;
    }
    if (!found) {
      tmp.push_back(v[i]);
    }
  }

  v = tmp;
  return;
}

// ------------------




void SineModelAnal::configure() {

  std::string orderBy = parameter("orderBy").toLower();
  if (orderBy == "magnitude") {
    orderBy = "amplitude";
  }
  else if (orderBy == "frequency") {
    orderBy = "position";
  }
  else {
    throw EssentiaException("Unsupported ordering type: '" + orderBy + "'");
  }

  Real maxFrequency  = std::min(float(parameter("sampleRate").toReal()/2.0), float(parameter("maxFrequency").toReal()));

  _peakDetect->configure("interpolate", true,
                         "range", parameter("sampleRate").toReal()/2.0,
                         "maxPeaks", parameter("maxPeaks"),
                         "minPosition", parameter("minFrequency"),
                         "maxPosition", maxFrequency,
                         "threshold", parameter("magnitudeThreshold"),
                         "orderBy", orderBy);


}



void SineModelAnal::compute() {
  // inputs and outputs
  const std::vector<std::complex<Real> >& fft = _fft.get();

  std::vector<Real>& tpeakMagnitude = _magnitudes.get();
  std::vector<Real>& tpeakFrequency = _frequencies.get();
  std::vector<Real>& tpeakPhase = _phases.get();

  // temp arrays
  std::vector<Real> peakMagnitude;
  std::vector<Real> peakFrequency;
  std::vector<Real> peakPhase;


  std::vector<Real> fftmag;
  std::vector<Real> fftmagdB;
  std::vector<Real> fftphase;

  _cartesianToPolar->input("complex").set(fft);
  _cartesianToPolar->output("magnitude").set(fftmag);
  _cartesianToPolar->output("phase").set(fftphase);

  _peakDetect->input("array").set(fftmagdB);
  _peakDetect->output("positions").set(peakFrequency);
  _peakDetect->output("amplitudes").set(peakMagnitude);

  _cartesianToPolar->compute();

  // convert to dB
    for (int i=0; i < (int)fftmag.size(); ++i){
      fftmagdB.push_back(20.f * std::log10(fftmag[i] + 1e-10));
    }

  _peakDetect->compute();

  phaseInterpolation(fftphase, peakFrequency, peakPhase);

  // tracking
  sinusoidalTracking(peakMagnitude, peakFrequency, peakPhase, _lasttpeakFrequency, parameter("freqDevOffset").toReal(), parameter("freqDevSlope").toReal(), tpeakMagnitude, tpeakFrequency, tpeakPhase);



  // limit number of tracks to maxnSines
  int maxSines = int ( parameter("maxnSines").toReal() );

  tpeakFrequency.resize(maxSines);
  tpeakMagnitude.resize(maxSines);
  tpeakPhase.resize(maxSines);

  // keep last frequency peaks for tracking
  _lasttpeakFrequency = tpeakFrequency;

}


// ---------------------------
// additional methods

void SineModelAnal::sinusoidalTracking(std::vector<Real>& peakMags, std::vector<Real>& peakFrequencies, std::vector<Real>& peakPhases, const std::vector<Real> tfreq, Real freqDevOffset, Real freqDevSlope, std::vector<Real> &tmagn, std::vector<Real> &tfreqn, std::vector<Real> &tphasen ){

  //	pfreq, pmag, pphase: frequencies and magnitude of current frame
  //	tfreq: frequencies of incoming tracks from previous frame
  //	freqDevOffset: minimum frequency deviation at 0Hz
  //	freqDevSlope: slope increase of minimum frequency deviation


  // sort current peaks per magnitude

  // -----
  // init arrays
  tfreqn.resize (tfreq.size());
  std::fill(tfreqn.begin(), tfreqn.end(), 0.);
  tmagn.resize (tfreq.size());
  std::fill(tmagn.begin(), tmagn.end(), 0.);
  tphasen.resize (tfreq.size());
  std::fill(tphasen.begin(), tphasen.end(), 0.);

  //	pindexes = np.array(np.nonzero(pfreq), dtype=np.int)[0]    # indexes of current peaks
  std::vector<int> pindexes;
  for (int i=0;i < (int)peakFrequencies.size(); ++i){  if (peakFrequencies[i] > 0) pindexes.push_back(i); }

  //	incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[0] # indexes of incoming tracks
  std::vector<Real> incomingTracks ;
  for (int i=0;i < (int)tfreq.size(); ++i){ if (tfreq[i]>0) incomingTracks.push_back(i); }
  //	newTracks = np.zeros(tfreq.size, dtype=np.int) -1           # initialize to -1 new tracks
  std::vector<int> newTracks(tfreq.size());
  std::fill(newTracks.begin(), newTracks.end(), -1);

  //	magOrder = np.argsort(-pmag[pindexes])                      # order current peaks by magnitude
  std::vector<int> magOrder;
  sort_indexes(magOrder, peakMags, false);


  // copy temporary arrays (as reference)
  std::vector<Real>	&pfreqt = peakFrequencies;
  std::vector<Real>	&pmagt = peakMags;
  std::vector<Real>	&pphaset = peakPhases;


  // -----
  // loop for current peaks

  if (incomingTracks.size() > 0 ){
    int i,j;
    int closestIdx;
    Real freqDistance;


    for (j=0; j < (int)magOrder.size() ; j++) {
      i = magOrder[j]; // sorted peak index
      if (incomingTracks.size() == 0)
      break; // all tracks have been processed

      // find closest peak to incoming track
      closestIdx = 0;
      freqDistance = 1e10;
      for (int k=0; k < (int)incomingTracks.size(); ++k){

        if (freqDistance > std::abs(pfreqt[i] - tfreq[incomingTracks[k]])){
          freqDistance = std::abs(pfreqt[i] - tfreq[incomingTracks[k]]);
          closestIdx = k;
        }
      }
      if (freqDistance < (freqDevOffset + freqDevSlope * pfreqt[i])) //  # choose track if distance is small
      {
        newTracks[incomingTracks[closestIdx]] = i;     //     # assign peak index to track index
        incomingTracks.erase(incomingTracks.begin() + closestIdx);              // # delete index of track in incomming tracks
      }
    }
  }


  //	indext = np.array(np.nonzero(newTracks != -1), dtype=np.int)[0]   # indexes of assigned tracks
  std::vector<int> indext;
  for (int i=0; i < (int)newTracks.size(); ++i)
  {
    if (newTracks[i] != -1) indext.push_back(i);
  }
  //	if indext.size > 0:
  if (indext.size() > 0)
  {
    //		indexp = newTracks[indext]                                    # indexes of assigned peaks
    std::vector<int> indexp;
    copy_int_vector_from_indexes(indexp, newTracks, indext);

    for (int i=0; i < (int)indexp.size(); ++i){
      tfreqn[indext[i]] = pfreqt[indexp[i]];                           //    # output freq tracks
      tmagn[indext[i]] = pmagt[indexp[i]];                             //    # output mag tracks
      tphasen[indext[i]] = pphaset[indexp[i]];                         //    # output phase tracks
    }

    // delete used peaks
    erase_vector_from_indexes(pfreqt, indexp);
    erase_vector_from_indexes(pmagt, indexp);
    erase_vector_from_indexes(pphaset, indexp);
  }

  // -----
  // create new tracks for non used peaks
  std::vector<int> emptyt;
  for (int i=0; i < (int)tfreq.size(); ++i)
  {
    if (tfreq[i] == 0) emptyt.push_back(i);
  }

  //	peaksleft = np.argsort(-pmagt)                                  # sort left peaks by magnitude
  std::vector<int> peaksleft;
  sort_indexes(peaksleft, pmagt, false);

  if ((peaksleft.size() > 0) && (emptyt.size() >= peaksleft.size())){    // fill empty tracks

    for (int i=0;i < (int)peaksleft.size(); i++)
    {
      tfreqn[emptyt[i]] = pfreqt[peaksleft[i]];
      tmagn[emptyt[i]] = pmagt[peaksleft[i]];
      tphasen[emptyt[i]] = pphaset[peaksleft[i]];
    }
  }
  else
  {
    if  ((peaksleft.size() > 0) && (emptyt.size() < peaksleft.size())) { //  add more tracks if necessary

      for (int i=0;i < (int)emptyt.size(); i++)
      {
        tfreqn[emptyt[i]] = pfreqt[peaksleft[i]];
        tmagn[emptyt[i]] = pmagt[peaksleft[i]];
        tphasen[emptyt[i]] = pphaset[peaksleft[i]];
      }
      for (int i=(int)emptyt.size();i < (int)peaksleft.size(); i++)
      {
        tfreqn.push_back(pfreqt[peaksleft[i]]);
        tmagn.push_back(pmagt[peaksleft[i]]);
        tphasen.push_back(pphaset[peaksleft[i]]);
      }
    }
  }


}



void SineModelAnal::phaseInterpolation(std::vector<Real> fftphase, std::vector<Real> peakFrequencies, std::vector<Real>& peakPhases){

  int N = peakFrequencies.size();
  peakPhases.resize(N);

  int idx;
  float  a, pos;
  int fftSize = fftphase.size();

  for (int i=0; i < N; ++i){
    // linear interpolation. (as done in numpy.interp function)
    pos =  fftSize * (peakFrequencies[i] / (parameter("sampleRate").toReal()/2.0) );
    idx = int ( 0.5 + pos ); // closest index

    a = pos - idx; // interpolate factor
    // phase diff smaller than PI to do intperolation and avoid jumps
    if (a < 0 && idx > 0){
      peakPhases[i] =  (std::abs(fftphase[idx-1] - fftphase[idx]) < Real(M_PI)) ? a * fftphase[idx-1] + (1.0 -a) * fftphase[idx] : fftphase[idx];
    }
    else {
      if (idx < fftSize-1 ){
        peakPhases[i] = (std::abs(fftphase[idx+1] - fftphase[idx]) < Real(M_PI)) ? a * fftphase[idx+1] + (1.0 -a) * fftphase[idx]: fftphase[idx];
      }
      else {
       peakPhases[i] = fftphase[idx];
     }
    }
  }
}

