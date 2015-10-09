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

#include "spsmodelanal.h"
#include "essentiamath.h"

using namespace essentia;
using namespace standard;

const char* SpsModelAnal::name = "SpsModelAnal";
const char* SpsModelAnal::description = DOC("This algorithm computes the stochastic model analysis. \n"
"\n"
"It is recommended that the input \"spectrum\" be computed by the Spectrum algorithm. This algorithm uses SineModelAnal. See documentation for possible exceptions and input requirements on input \"spectrum\".\n"
"\n"
"References:\n"
"  https://github.com/MTG/sms-tools\n"
"  http://mtg.upf.edu/technologies/sms\n"
);

/*
// ------------------
// Additional support functions
typedef std::pair<int,Real> mypair;
// sort indexes: get arguments of sorted vector
bool comparator_up ( const mypair& l, const mypair& r)
{
  return l.second < r.second;
}
bool comparator_down ( const mypair& l, const mypair& r)
{
  return l.second > r.second;
}

// It sorts the indexes of an input vector v, and outputs the sorted index vector idx
void SpsModelAnal::sort_indexes(std::vector<int> &idx, const std::vector<Real> &v, bool ascending) {

  // initialize original index locations
  std::vector<mypair> pairs(v.size());
  for (int i = 0; i != pairs.size(); ++i){
    pairs[i].first = i;
    pairs[i].second = v[i];
  }

  // sort indexes based on comparing values in v
  if (ascending)
    sort(pairs.begin(), pairs.end(),comparator_up);
  else
    sort(pairs.begin(), pairs.end(),comparator_down);

  // copy sorted indexes
  for (int i = 0; i != pairs.size(); ++i) idx.push_back(pairs[i].first);

  return;
}

void SpsModelAnal::copy_vector_from_indexes(std::vector<Real> &out, const std::vector<Real> v, const std::vector<int> idx){

  for (int i = 0; i < idx.size(); ++i){
    out.push_back(v[idx[i]]);
  }
  return;
}

void SpsModelAnal::copy_int_vector_from_indexes(std::vector<int> &out, const std::vector<int> v, const std::vector<int> idx){

  for (int i = 0; i < idx.size(); ++i){
    out.push_back(v[idx[i]]);
  }
  return;
}

// erase elements from a vector given a vector of indexes
void SpsModelAnal::erase_vector_from_indexes(std::vector<Real> &v, const std::vector<int> idx){
  std::vector<Real> tmp;
  bool found;
  for (int i = 0; i < v.size(); ++i) {
    found = false;
    for (int j = 0; j < idx.size(); ++j){
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

*/
// ------------------




void SpsModelAnal::configure() {

//  std::string orderBy = parameter("orderBy").toLower();
//  if (orderBy == "magnitude") {
//    orderBy = "amplitude";
//  }
//  else if (orderBy == "frequency") {
//    orderBy = "position";
//  }
//  else {
//    throw EssentiaException("Unsupported ordering type: '" + orderBy + "'");
//  }

//  _peakDetect->configure("interpolate", true,
//                         "range", parameter("sampleRate").toReal()/2.0,
//                         "maxPeaks", parameter("maxPeaks"),
//                         "minPosition", parameter("minFrequency"),
//                         "maxPosition", parameter("maxFrequency"),
//                         "threshold", parameter("magnitudeThreshold"),
//                         "orderBy", orderBy);
_sineModelAnal->configure( "sampleRate", parameter("sampleRate").toReal(),
                            "maxnSines", parameter("maxnSines").toReal() ,
                            "freqDevOffset", parameter("freqDevOffset").toReal(),
                            "freqDevSlope",  parameter("freqDevSlope").toReal()
                            );

_sineModelSynth->configure( "sampleRate", parameter("sampleRate").toReal(),
                            "fftSize", parameter("frameSize").toReal(),
                            "hopSize", parameter("hopSize").toReal()
                            );

}



void SpsModelAnal::compute() {
  // inputs and outputs
  const std::vector<std::complex<Real> >& fft = _fft.get();

  std::vector<Real>& peakMagnitude = _magnitudes.get();
  std::vector<Real>& peakFrequency = _frequencies.get();
  std::vector<Real>& peakPhase = _phases.get();
  std::vector<Real>& stocEnv = _stocenv.get();

  std::vector<Real> fftmag;
  std::vector<Real> fftphase;

 _sineModelAnal->input("complex").set(fft);
 _sineModelAnal->output("magnitudes").set(peakMagnitude);
 _sineModelAnal->output("frequencies").set(peakFrequency);
 _sineModelAnal->output("phases").set(peakPhase);

  _sineModelAnal->compute();

  // compute stochastic envelope
  stochasticModelAnal(fft, peakMagnitude, peakFrequency, peakPhase, &stocEnv);

}


// ---------------------------
// additional methods


void spsModelAnal::stochasticModelAnal(const std::vector<std::complex<Real> > fftInput, const std::vector<Real> magnitudes, const std::vector<Real> frequencies, const std::vector<Real> phases, std::vector<Real> &stocEnv)
{

  // subtract sines
  std::vector<std::complex<Real> > fftSines;
  std::vector<std::complex<Real> > fftRes;

  _sineModelSynth->input("magnitudes").set(magnitudes);
  _sineModelSynth->input("frequencies").set(frequencies);
  _sineModelSynth->input("phases").set(phases);
  _sineModelSynth->output("fft").set(fftSines);

  _sineModelSynth->compute();

  fftRes = fftInput; // initialize output

  for (int i= 0; i < (int)fftRes.size(); ++i)
  {
    fftRes[i].real(fftRes[i].real() - fftSines[i].real());
    fftRes[i].imag(fftRes[i].imag() - fftSines[i].imag());
  }

  // compute residual envelope
/*
		mXr = 20*np.log10(abs(Xr[:hN]))                     # magnitude spectrum of residual
		mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)  # decimate the mag spectrum
		if l == 0:                                          # if first frame
			stocEnv = np.array([mXrenv])
		else:                                               # rest of frames
			stocEnv = np.vstack((stocEnv, np.array([mXrenv])))
		pin += H                                            # advance sound pointer
	return stocEnv
	*/

}


/*
void SpsModelAnal::sinusoidalTracking(std::vector<Real>& peakMags, std::vector<Real>& peakFrequencies, std::vector<Real>& peakPhases, const std::vector<Real> tfreq, Real freqDevOffset, Real freqDevSlope, std::vector<Real> &tmagn, std::vector<Real> &tfreqn, std::vector<Real> &tphasen ){

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
  for (int i=0;i < peakFrequencies.size(); ++i){  if (peakFrequencies[i] > 0) pindexes.push_back(i); }

  //	incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[0] # indexes of incoming tracks
  std::vector<Real> incomingTracks ;
  for (int i=0;i < tfreq.size(); ++i){ if (tfreq[i]>0) incomingTracks.push_back(i); }
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


    for (j=0; j < magOrder.size() ; j++) {
      i = magOrder[j]; // sorted peak index
      if (incomingTracks.size() == 0)
      break; // all tracks have been processed

      // find closest peak to incoming track
      closestIdx = 0;
      freqDistance = 1e10;
      for (int k=0; k < incomingTracks.size(); ++k){

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
  for (int i=0; i < newTracks.size(); ++i)
  {
    if (newTracks[i] != -1) indext.push_back(i);
  }
  //	if indext.size > 0:
  if (indext.size() > 0)
  {
    //		indexp = newTracks[indext]                                    # indexes of assigned peaks
    std::vector<int> indexp;
    copy_int_vector_from_indexes(indexp, newTracks, indext);

    for (int i=0; i < indexp.size(); ++i){
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
  for (int i=0; i < tfreq.size(); ++i)
  {
    if (tfreq[i] == 0) emptyt.push_back(i);
  }

  //	peaksleft = np.argsort(-pmagt)                                  # sort left peaks by magnitude
  std::vector<int> peaksleft;
  sort_indexes(peaksleft, pmagt, false);

  if ((peaksleft.size() > 0) && (emptyt.size() >= peaksleft.size())){    // fill empty tracks

    for (int i=0;i < peaksleft.size(); i++)
    {
      tfreqn[emptyt[i]] = pfreqt[peaksleft[i]];
      tmagn[emptyt[i]] = pmagt[peaksleft[i]];
      tphasen[emptyt[i]] = pphaset[peaksleft[i]];
    }
  }
  else
  {
    if  ((peaksleft.size() > 0) && (emptyt.size() < peaksleft.size())) { //  add more tracks if necessary

      for (int i=0;i < emptyt.size(); i++)
      {
        tfreqn[emptyt[i]] = pfreqt[peaksleft[i]];
        tmagn[emptyt[i]] = pmagt[peaksleft[i]];
        tphasen[emptyt[i]] = pphaset[peaksleft[i]];
      }
      for (int i=emptyt.size();i < peaksleft.size(); i++)
      {
        tfreqn.push_back(pfreqt[peaksleft[i]]);
        tmagn.push_back(pmagt[peaksleft[i]]);
        tphasen.push_back(pphaset[peaksleft[i]]);
      }
    }
  }
}



void SpsModelAnal::phaseInterpolation(std::vector<Real> fftphase, std::vector<Real> peakFrequencies, std::vector<Real>& peakPhases){

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
      peakPhases[i] =  (std::abs(fftphase[idx-1] - fftphase[idx]) > Real(M_PI)) ? a * fftphase[idx-1] + (1.0 -a) * fftphase[idx] : fftphase[idx];
    }
    else {
      if (idx < fftSize-1 ){
        peakPhases[i] = (std::abs(fftphase[idx+1] - fftphase[idx]) > Real(M_PI)) ? a * fftphase[idx+1] + (1.0 -a) * fftphase[idx]: fftphase[idx];
      }
      else {
       peakPhases[i] = fftphase[idx];
     }
    }
  }
}

*/
