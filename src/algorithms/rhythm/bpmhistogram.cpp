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

#include <cassert>
#include "bpmhistogram.h"
#include "essentiamath.h"
#include "bpmutil.h"
#include <cassert>
#include "tnt/tnt2vector.h"

using namespace std;
using namespace essentia;


#include "poolstorage.h"

namespace essentia {
namespace streaming {

const char* BpmHistogram::name = "BpmHistogram";
const char* BpmHistogram::description = DOC("Given the novelty curve (see NoveltyCurve algorithm), this algorithm outputs a histogram of the most probable bpms and their magnitudes as a measure of strength. It also outputs the mean of the strongest bpm present in the signal. In addition it outputs the bpm at each frame which is most similar to the mean bpm, a half-wave rectified sinusoid whose peaks represent the ticks of the audio signal and their amplitude.\n"
"The sampleRate parameter refers to the framerate at which the novelty curve has been computed, thus the audio sampling rate divided by the hopsize at which the audio signal was processed.\n"
"The outputs of the algorithm are the following: \n"
" - bpm: is the mean of the most salient bpm.\n"
" - bpmCandidates: list of the strongest bpms present in the signal.\n"
" - bpmMagnitudes: list containing the normalized strength of each of the bpms from the previous output. These two outputs can be used to construct a histogram and take your own decision when mean bpm is wrong\n."
" - tempogram: kind of a spectrogram indexed by bpm where the value at each index is the magnitude of the bpm. Very useful for detecting tempo variations and for plotting the evolution of tempi.\n"
" - frameBpms: list containing the candidate bpms at each frame that are most similar to the meanBpm. If no candidates are found to be similar to the mean bpm, the meanBpm will be kept unless \"tempoChange\" seconds have triggered a variation in the tempo.\n"
" - ticks: outputs the ticks' positions in seconds.\n"
" - ticksMagnitude: returns the magnitude of each tick. The higher value the higher probabylity to be correct.\n"
" - sinusoid: outputs a sinusoidal model of the tick's positions. The previous outputs are based on detecting the peaks of this half-wave rectified sinusoid. If needed, one should be able to drive its own peak detection algorithm on this sinusoid in order to obtain its own ticks. Beware that due to overlap factors the last few ticks may exceed the length of the audio signal. Therfore, this output should always be checked against the length of the audio signal.\n"

"Although the algorithm tries to find the beats that best fit to the mean bpm, the tempo is not assumed to be constant unless specified in the corresponding parameter.  For this reason and if the tempo differs too much from frame to frame, there may be phase discontinuities when constructing the sinusoid which can yield to too many ticks. When this occurs, one can use the sinusoid output to recursively run this algorithm until the ticks stabilize. At this point it may be useful to induce/infer a specific bpm and set the constant tempo parameter to true.\n"
"Another useful trick, is to run the algorithm one time to get an estimation of the bpm and rerun it with a frameSize parameter which is a multiple of the mean bpm.\n"
"\n"
"NOTE that using RhythmExtractor2013 is recommended in order to extract beats, as it was found to perform better in evaluations.\n"
"\n"
"Quality: outdated (use RhythmExtractor2013 instead, still this algorithm might be useful when working with other onset detection functions apart from NoveltyCurve)\n"
"\n"
"References:\n"
"  [1] P. Grosche and M. Müller, \"A mid-level representation for capturing\n"
"  dominant tempo and pulse information in music recordings,\" in\n"
"  International Society for Music Information Retrieval Conference\n"
"  (ISMIR’09), 2009, pp. 189–194.");


BpmHistogram::BpmHistogram() : _normalize(false), _weightByMagnitude(false) {

  declareInput(_signal, "novelty", "the novelty curve");

  declareOutput(_bpm, 0, "bpm", "the mean of the most salient tempo");
  declareOutput(_bpmCandidates, 0, "bpmCandidates", "the list of bpm candidates");
  declareOutput(_bpmMagnitudes, 0, "bpmMagnitudes", "the strength of bpm candidates");
  declareOutput(_tempogram, 0, "tempogram", "the bpm spectrogram");
  declareOutput(_frameBpms, 0, "frameBpms", "the bpm at each frame that is most related to the mean bpm");
  declareOutput(_ticks, 0, "ticks", "the list of ticks' positions [s]");
  declareOutput(_ticksMagnitude, 0, "ticksMagnitude", "the strength of the ticks");
  declareOutput(_sinusoid, 0, "sinusoid", "the sinusoid whose peaks indicate the ticks' positions");


  // streaming algos:
  AlgorithmFactory& factory = AlgorithmFactory::instance();
  _frameCutter   = factory.create("FrameCutter");
  _windowing     = factory.create("Windowing");
  _fft           = factory.create("FFT");
  _cart2polar    = factory.create("CartesianToPolar");
  _peakDetection = factory.create("PeakDetection");


  // Connect internal algorithms
  _signal                               >>   _frameCutter->input("signal");
  _frameCutter->output("frame")         >>   _windowing->input("frame");
  _windowing->output("frame")           >>   _fft->input("frame");
  _fft->output("fft")                   >>   _cart2polar->input("complex");
  _cart2polar->output("magnitude")      >>   _peakDetection->input("array");
  _cart2polar->output("magnitude")      >>   PC(_pool, "magnitudes");
  _cart2polar->output("phase")          >>   PC(_pool, "phases");
  _peakDetection->output("amplitudes")  >>   PC(_pool, "peaks_value");
  _peakDetection->output("positions")   >>   PC(_pool, "peaks_positions");

  _network = new scheduler::Network(_frameCutter);
}

BpmHistogram::~BpmHistogram() {
  delete _network;
}

void BpmHistogram::configure() {
  _frameRate = parameter("frameRate").toReal();
  _frameSize = int(parameter("frameSize").toReal()*_frameRate);
  _frameSize = nextPowerTwo(int(ceil(Real(_frameSize))));
  _hopSize = int(_frameSize/parameter("overlap").toReal());
  int zeroPadding = parameter("zeroPadding").toInt()*_frameSize;
  _binWidth = _frameRate/(_frameSize+zeroPadding);
  _maxPeaks = parameter("maxPeaks").toInt();
  _minBpm = floor(parameter("minBpm").toReal());
  _maxBpm = ceil(parameter("maxBpm").toReal());
  _weightByMagnitude = parameter("weightByMagnitude").toBool();
  _bpmTolerance = 3; // 3%
  _constantTempo = parameter("constantTempo").toBool();
  _meanBpm = parameter("bpm").toReal();

  Real minBin = _minBpm/(_binWidth*60);
  Real maxBin = _maxBpm/(_binWidth*60);

  _frameCutter->configure("frameSize", _frameSize,
                          "hopSize", _hopSize,
                          "silentFrames", "keep",
                          "validFrameThresholdRatio", 0.5,
                          "startFromZero", false);

  _windowing->configure("type", parameter("windowType"), "zeroPadding", zeroPadding, "zeroPhase", true);

  _peakDetection->configure("orderBy", "amplitude",
                            "range", (_frameSize+zeroPadding)/2, //+1,  // not adding 1 due to how the algorithm works
                            "maxPeaks", _maxPeaks,
                            "interpolate", true,
                            "threshold", 0,
                            "minPosition", minBin,
                            "maxPosition", maxBin);

  createWindow(_frameSize); // create a window for overlap-&-add the final sinusoids
}

void BpmHistogram::createWindow(int size) {
  standard::Algorithm* windowing = standard::AlgorithmFactory::create("Windowing",
                                                                      "zeroPhase", false,
                                                                      "type",parameter("windowType"));
  vector<Real> ones(size, 1.0);
  windowing->input("frame").set(ones);
  windowing->output("frame").set(_window);
  windowing->compute();
  delete windowing;
  essentia::normalize(_window);
}

void BpmHistogram::computeBpm() {
  const vector<vector<Real> >& magnitudes = _pool.value<vector<vector<Real> > >("magnitudes");
  const vector<vector<Real> >& peaks = _pool.value<vector<vector<Real> > >("peaks_positions");
  const vector<vector<Real> >& peaksValue = _pool.value<vector<vector<Real> > >("peaks_value");
  Real bpmRatio = _binWidth*60.0;
  Real threshold = 0;
  //TODO: scheduling problem!! seems that there are more vectors in magnitudes than peaks
  //for (int i=0; i<(int)magnitudes.size();i++) {
  for (int i=0; i<(int)peaks.size();i++) {
    vector<Real> tempogram(int(_maxBpm+1), Real(0));
    try {
      //threshold = max(Real(1e-4), max(median(peaksValue), mean(peaksValue))); // only use peaks that are VERY prominent
      threshold = min(Real(1e-6), min(median(magnitudes[i]), mean(magnitudes[i]))); // be permissive
      //threshold = max(Real(1e-4), min(median(peaksValue), mean(peaksValue)));
    }
    catch(const EssentiaException& ) { // no peaks found
      threshold = numeric_limits<int>::max();
    }
    vector<Real> mainPeaks, mainBpms;
    mainPeaks.reserve(peaks[i].size());
    mainBpms.reserve(peaks[i].size());
    for (int j=0; j<(int)peaks[i].size(); j++) {
      if (peaksValue[i][j] < threshold) continue;
      Real bpm = round(peaks[i][j]*bpmRatio);
      // as peakdetection minPosition is rounded by bins we must double check
      // that we don't get a bpm outside the parameter ranges
      if (bpm > _maxBpm || bpm < _minBpm) continue;
      mainPeaks.push_back(peaks[i][j]);
      mainBpms.push_back(bpm);
      _pool.add("bpmCandidates", bpm);
      _pool.add("bpmAmplitudes", peaksValue[i][j]);
      tempogram[int(bpm)] = peaksValue[i][j];
    }
    // check for silent frame or dc offset or possible constant input. Normally more than one peak should be found
    if (mainPeaks.size() < 1) {
      mainPeaks.clear();
      mainBpms.clear();
      _pool.add("magnitudes", vector<Real>(magnitudes[i].size(), 0));
      _pool.add("bpmCandidates", 0);
      _pool.add("bpmAmplitudes", 0);
    }
    _pool.add("tempogram", tempogram);
  }

}

void BpmHistogram::unwrapPhase(Real& ph, const Real& uwph) {
  Real diff = uwph - ph;
	if( fabs(diff) <= M_PI ) return;
	int k=0;
	if( diff < 0 ) k = int(diff/(M_2PI) -0.5 );//- 0.5 );
	else k = int(diff/(M_2PI) + 0.5 );
	ph += Real(k)*M_2PI;
}

void BpmHistogram::createTicks(Real bpm) { //const vector<Real>& bpms) {
  if (bpm==0) {
    // this could happen:
    // 1. if any of the found bpms were above or below the maxBpm/minBpm
    // 2. silent track
    return;
  }
  // TODO: what peaks should be taken for finding each frame's bpm?
  // prominent_peaks yield to worse alignment of sinusoids...
  //const vector<vector<Real> >& peaks = _pool.value<vector<vector<Real> > >("prominent_peaks_positions");
  const vector<vector<Real> >& peaks = _pool.value<vector<vector<Real> > >("peaks_positions");
  const vector<vector<Real> >& ph= _pool.value<vector<vector<Real> > >("phases");

  vector<vector<Real> > phases(ph.begin(), ph.end());
  for (int iFrame=0; iFrame<(int)phases.size(); iFrame++){
    Real uwph = phases[iFrame][0];
    for (int ibin=1; ibin<(int)phases[iFrame].size(); ibin++){
      unwrapPhase(phases[iFrame][ibin], uwph);
      uwph = phases[iFrame][ibin];
    }
  }

  Real bpmRatio = _binWidth*60.0;

  // for each frame find if there is a peak which is equal to the song's bpm,
  // if not equal then search for a harmonic of the song's bpm, else get the
  // maximum peak
  int nFrames = peaks.size();
  Real lastBin = -1;
  vector<Real> sinusoid(_frameCutter->output("frame").totalProduced()*_hopSize, Real(0.0));
  for (int iFrame=0; iFrame<nFrames; iFrame+=1) {
    Real bin = -1;
    if (peaks[iFrame].empty()){ // framecutter is set to not drop silent frames!
      bin = 0;
      _pool.add("frameBpms", 0);
      continue;
    }
    for (int j=0; j<int(peaks[iFrame].size()); j++) {
      Real bpmCandidate = peaks[iFrame][j]*bpmRatio;
      if (areEqual(bpm, bpmCandidate, _bpmTolerance)) {
          bin=peaks[iFrame][j];
          break;
      }
    }
    if (bin<0) {
      for (int j=0; j<int(peaks[iFrame].size()); j++) {
        Real bpmCandidate = peaks[iFrame][j]*bpmRatio;
        if (bpmCandidate < _minBpm || bpmCandidate > _maxBpm) continue;
        if (_constantTempo) {
          if (areHarmonics(bpm, bpmCandidate, _bpmTolerance, false) &&
              greatestCommonDivisor(bpm, bpmCandidate, _bpmTolerance) > _minBpm){
            Real ratio = bpm/bpmCandidate;
            if (ratio < 1) {
              ratio = 1.0/ratio;
              bin=peaks[iFrame][j]/round(ratio);
            }
            else bin=peaks[iFrame][j]*round(ratio);
            break;
          }
        }
        else {
          // here the greatestCommonDivisor allows tempo changes that are not integer
          // multiples of the mainBpm (i.e. 5/2, 4/3, etc.)
          if (areHarmonics(bpm, bpmCandidate, _bpmTolerance, false) ||
              greatestCommonDivisor(bpm, bpmCandidate, _bpmTolerance) > _minBpm){
            bin=peaks[iFrame][j];
            break;
          }
        }
      }
    }
    //if (bin>0) break;
    if (bin < 0) {
      for (int j=0; j<(int)peaks[iFrame].size(); j++) {
        Real bpmCandidate = peaks[iFrame][j]*bpmRatio;
        Real ratio = 0;
        if (lastBin>0) ratio = lastBin/peaks[iFrame][j];
        else ratio = bpm/bpmCandidate;
        if (ratio < 1) ratio = 1.0/ratio;
        if (ratio <= 2) {
          bin = peaks[iFrame][j];
          break;
        }
      }
    }
    if (bin < 0) bin=0;
    lastBin = bin;

    _pool.add("frameBpms", bin*bpmRatio);
  }

  vector<Real> frameBpms = _pool.value<vector<Real> >("frameBpms");
  postProcessBpms(bpm, frameBpms);
  Real maxBpm = 1;
  for (int iFrame=0; iFrame<nFrames; iFrame++) {
    // for click/pulse signals it works best to round to ints and not
    // unwrapping nor interpolating phase, on the other hand for real life is
    // the other way around
    Real bin = frameBpms[iFrame]/bpmRatio;
    if (frameBpms[iFrame] > maxBpm) maxBpm = frameBpms[iFrame];
    int intBin = int(bin);
    Real phase = phases[iFrame][intBin] + (bin-intBin)*(phases[iFrame][intBin+1]-phases[iFrame][intBin]);
    Real freq = bin*_binWidth;
    //cout << "t: " << iFrame << "\tt: " << iFrame*_hopSize/_frameRate
    //     << "\tframeBpm: " << frameBpms[iFrame]
    //     << "\tfreq: " << freq
    //     << "\tphase: " << phase
    //     << endl;
    if (freq > 0) {
    createSinusoid(sinusoid, freq, phase, iFrame);
    }
  }

  // TODO: we need a better smoothing!!!! and if possible without delay
  Real mavgSize = 30./maxBpm*_frameRate; //0.5*maxBpm in samples
  standard::Algorithm* mavg = standard::AlgorithmFactory::create("MovingAverage",
                                                                 "size", (int)mavgSize);
  vector<Real> sinusoid_ma;
  mavg->input("signal").set(sinusoid);
  mavg->output("signal").set(sinusoid_ma);
  mavg->compute();
  delete mavg;
  sinusoid.assign(sinusoid_ma.begin(), sinusoid_ma.end());

  standard::Algorithm* peakDetect= standard::AlgorithmFactory::create("PeakDetection",
                                                                      "orderBy", "position",
                                                                      "range", (int)sinusoid.size()-1,
                                                                      "maxPeaks", (int)sinusoid.size(),
                                                                      "interpolate", true,
                                                                      "threshold", 1e-4,
                                                                      "minPosition", 0,
                                                                      "maxPosition", (int)sinusoid.size()-1);
  vector<Real> ticks, ticksAmp;
  peakDetect->input("array").set(sinusoid);
  peakDetect->output("positions").set(ticks);
  peakDetect->output("amplitudes").set(ticksAmp);
  peakDetect->compute();
  delete peakDetect;

  // convert ticks to seconds
  //due to the moving average filter a delay should be taken into account.
  //Delaying only by half of the size, because the peak detection algorithm
  //outputs the middle position of a "plateau" and the sinusoid has been
  //converted into many plateaus after the moving average.
  Real delay = mavgSize/2.0 ;
  // novelty curve has also been smoothed by a moving average filter
  int delayFromNoveltyCurve = int(0.1*_frameRate);
  if (delayFromNoveltyCurve%2) delayFromNoveltyCurve++;
  delay += delayFromNoveltyCurve/2;


  for (int i=0; i<(int)ticks.size(); i++) {
    ticks[i] -= delay;
    if (ticks[i] < 0) ticks[i]=0;
    ticks[i]/=_frameRate;
  }

  _tempogram.push(vecvecToArray2D(_pool.value<vector<vector<Real> > >("tempogram")));
  _frameBpms.push(frameBpms);
  _ticks.push(ticks);
  _ticksMagnitude.push(ticksAmp);
  _sinusoid.push(sinusoid);
}

void BpmHistogram::createSinusoid(vector<Real>& sinusoid, Real freq, Real phase, int idx) {
  int size = _window.size();
  vector<Real> sine(size);
  int pos = (idx)*_hopSize;
  for (int i=0; i<size; i++) {
    if (pos+i<0) continue;
    if (pos+i >= int(sinusoid.size())) break;
    Real s = _window[i]*cos(M_2PI*freq*Real(i)/_frameRate+phase);
    if (s>0) sinusoid[pos+i] += s;
  }
}

void BpmHistogram::postProcessBpms(Real mainBpm, vector<Real>& bpms) {
  Real meanBpm = 0;
  if (_meanBpm == 0) { // no bpm induction
    int counts=0;
    for (int i=0; i<(int)bpms.size(); i++) {
      if (areEqual(mainBpm, bpms[i], _bpmTolerance)) {
        meanBpm += bpms[i];
        counts++;
      }
    }
    _meanBpm = meanBpm/Real(counts);
  }
  int frameCount = int(parameter("tempoChange").toReal()*_frameRate/Real(_hopSize));
  for (int i=0; i<(int)bpms.size(); i++) {
    if (areEqual(_meanBpm, bpms[i], _bpmTolerance)|| bpms[i] == 0) continue;
    // check if this tempo change has continuity or not
    int count = 0;
    int j = i+1;
    while (j<(int)bpms.size() && areEqual(bpms[i], bpms[j], _bpmTolerance && bpms[j] != 0)) {
      count++;
      j++;
    }
    if (count < frameCount) { // TODO: how many frames should we consider for bpm continuity?
      if (areHarmonics(bpms[i], _meanBpm, _bpmTolerance, false)) {
          Real ratio = bpms[i]/_meanBpm;
          if (ratio<1) {
            ratio = 1.0/ratio;
            bpms[i] = bpms[i]*round(ratio);
          }
          else bpms[i] = bpms[i]/round(ratio);
      }
      else bpms[i] = _meanBpm;
    }
    else i=j;
  }
}

Real BpmHistogram::deviationWeight(Real x, Real mu, Real sigma) {
  // the rationale behind this function is that when a bpm has a high deviation
  // (sigma) the contribution from neighbouring bpms should be penalized.
  // the bigger the deviation the higher the penalization
  if (x<_minBpm) return 0;
  sigma = sigma/10.0;
  Real k = 1/(x*sigma*sqrt(M_2PI));
  Real dev = log(mu/x);
  return (exp((-(dev*dev))/(k*k)));
}

#if 0
Real BpmHistogram::lognormal(Real x) {
  // this function emphasizes values around 120bpm
  // it's generally good for 4/4 but not so good for other measures
  //Real sigma = 0.5;
  Real sigmaSqr = 0.25;
  Real mu = 150;
  Real dev = log(x/mu);
  Real k = 1.2533141373155001*x; // k = 1/(x*sigma*sqrt(M_2PI));
  k = 1.0/k;
  Real normk = 0.00602744916631;
  return k/normk*exp((-(dev*dev))/(2*sigmaSqr));
}
#endif

void BpmHistogram::computeHistogram(vector<Real>& bpmPositions,
                                    vector<Real>& bpmMagnitudes) {
  const vector<Real>& bpms = _pool.value<vector<Real> >("bpmCandidates");
  const vector<vector<Real> >& tempogram = _pool.value<vector<vector<Real> > >("tempogram");
  const vector<Real>& m = _pool.value<vector<Real> >("bpmAmplitudes");
  vector<Real> mags(m.begin(), m.end());
  essentia::normalize(mags);
  //Real maxBpm = *max_element(bpms.begin(), bpms.end());
  vector<Real> bpmHist = vector<Real>(int(_maxBpm+1), Real(0));

  // when building the histogram each contribution to a bpm-bin will be weighted by
  // its deviation to the bin. Thus a bpm of 60 will have a weight of 1 to the bin
  // number 60, whereas a bpm of 62 will have a weight (contribution) less than 1
  // to the bin number 60 (but 1 to the bin number 62)
  // TODO: Actually, we could get rid of this weighted histogram and just
  // compute a normal one.
  for (int i=0; i<(int)bpms.size(); i++)  {
    int pos = int(bpms[i]+0.5);
    if (pos == 0) continue;
    for (int j=0; j<(int)bpms.size(); j++) {
      Real candidate = round(bpms[j]);
      if (candidate==0) continue;
      if (areEqual(bpms[i], bpms[j], _bpmTolerance)) {
        Real weight = deviationWeight(candidate, pos, fabs(pos-candidate)); // maxDeviation[pos]);
        if (_weightByMagnitude) bpmHist[pos]+= weight*mags[j];
        else bpmHist[pos]+=weight;
      }
    }
  }

  // get peaks from histogram:
  int size = bpmHist.size();
  vector<Real> tmpBpms;
  tmpBpms.reserve(size);
  mags.clear();
  mags.reserve(size);

  Real accum = 0;
  int nonZeroValues = 0;
  for (int i=0; i<size; i++) {
    if (bpmHist[i] > 0) {
      accum+=bpmHist[i];
      nonZeroValues++;
    }
  }
  Real threshold = 0;
  if (nonZeroValues != 0) threshold = accum/nonZeroValues;

  while (tmpBpms.empty()) {
    if (threshold == 0) {
      tmpBpms.push_back(0.0);
      mags.push_back(0.0);
      break;
    }
    for (int i=0; i<size; i++) {
      if (bpmHist[i] >= threshold) {
        tmpBpms.push_back(i);
        mags.push_back(bpmHist[i]);
      }
    }
    sortpair<Real, Real, greater<Real> >(mags, tmpBpms);
    threshold *=0.5;
  }

  // At this point we should have a reasonable amount of bpm candidates, which
  // will be ordered by their energy in the tempogram

  //Real totalEnergy=0;
  for (int i=0; i<(int)tempogram.size(); i++) {
    Real totalEnergy = energy(tempogram[i]);
    if (totalEnergy == 0) continue;
    for (int j=0; j < (int)tmpBpms.size(); j++) {
      int start = int(max(Real(0), tmpBpms[j]-_bpmTolerance));
      int end = int(min(Real(tempogram[i].size()-1), tmpBpms[j]+_bpmTolerance));
      Real value = 0;
      for (int k=start; k<=end; k++) {
        value+=tempogram[i][k]*tempogram[i][k];
      }
      mags[j] += value/totalEnergy;
    }
  }
  sortpair<Real, Real, greater<Real> >(mags, tmpBpms);
  normalize(mags);
  bpmPositions.reserve(mags.size());
  for (int i=0; i<(int)mags.size(); i++) {
    if (mags[i] >=0.25) {
      bpmPositions.push_back(tmpBpms[i]);
      bpmMagnitudes.push_back(mags[i]);
    }
  }
  // merging is generally bad, as this may round the correct bpm to a different
  // bpm which will cause ticks not to be as exact. Tick errors will
  // unfortunately happen anyways, but merging may increase these errors.
  //mergeBpms(tmpBpms, mag, bpmTolerance);
}

AlgorithmStatus BpmHistogram::process() {
  if (!shouldStop()) return PASS;

  //compute bpm histogram:
  computeBpm();

  const vector<string>& descriptors = _pool.descriptorNames();
  if (!contains(descriptors, "bpmCandidates") ||
      sum(_pool.value<vector<Real> >("bpmCandidates")) == 0) {

    // silent track
    vector<Real> zero(1,0);
    vector<Real> empty;
    TNT::Array2D<Real> emptyMatrix;
    _bpm.push((Real)0.0);
    _bpmCandidates.push(empty);
    _bpmMagnitudes.push(empty);
    _tempogram.push(emptyMatrix);
    _frameBpms.push(empty);
    _ticks.push(empty);
    _ticksMagnitude.push(empty);
    _sinusoid.push(empty);;

    return FINISHED;
  }


  // get peaks from histogram:
  vector<Real> bpmPositions, bpmMagnitudes;
  computeHistogram(bpmPositions, bpmMagnitudes);

  if (_meanBpm != 0) createTicks(_meanBpm); // bpm induction
  else createTicks(bpmPositions[0]);

  normalize(bpmMagnitudes);
  _bpm.push(_meanBpm);
  _bpmCandidates.push(bpmPositions);
  _bpmMagnitudes.push(bpmMagnitudes);

  return FINISHED;
}



} // namespace streaming
} // namespace essentia
