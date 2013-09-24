
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

#ifndef ESSENTIA_BPMUTILS_H
#define ESSENTIA_BPMUTILS_H

#include "../essentiamath.h"
#include <cassert>

namespace essentia {

inline
Real lagToBpm(Real lag, Real sampleRate, Real hopSize) {
  return 60.0*sampleRate/lag/hopSize;
}

inline
Real bpmToLag(Real bpm, Real sampleRate, Real hopSize) {
  return lagToBpm(bpm, sampleRate, hopSize);
}

inline
int longestChain(const std::vector<Real>& dticks, int startpos, Real period, Real tolerance) {
  int pos = startpos;
  Real ubound = period*(1+tolerance);
  Real lbound = period*(1-tolerance);
  while ((pos < (int)dticks.size()) &&
         (lbound < dticks[pos] && dticks[pos] < ubound)) {
    pos++;
  }

  return pos - startpos;
}

inline
void bpmDistance(Real x, Real y, Real& error, Real& ratio) {
  ratio = x/y;
  error = -1;
  if (ratio < 1) {
    ratio = round(1./ratio);
    error=(x*ratio-y)/std::min(y,Real(x*ratio))*100;
  }
  else {
    ratio = round(ratio);
    error = (x-y*ratio)/std::min(x,Real(y*ratio))*100;
  }
}

inline
bool areEqual(Real a, Real b, Real tolerance) {
  //return fabs(a-b) <= epsilon;
  Real error=0;
  Real ratio=0;
  bpmDistance(a,b,error,ratio);
  return (std::fabs(error)<tolerance) && (int(ratio)==1);
}

inline
bool areHarmonics(Real x, Real y, Real epsilon, bool bPower2) {
  // epsilon must be in %. a strict choice could be 3
  Real ratio = 0;
  Real error = 0;
  bpmDistance(x, y, error, ratio);
  error = std::fabs(error);
  if (error <= epsilon) {
    if (bPower2) return isPowerTwo(int(fabs(ratio)));
    return true;
  }
  return false;
}

inline
Real greatestCommonDivisor(Real x, Real y, Real epsilon) {
  // epsilon must be in %. a strict choice could be 3
  if (x<y) return greatestCommonDivisor(y,x,epsilon);
  if (x==0) return 0;
  Real error = std::numeric_limits<int>::max(),
  ratio=std::numeric_limits<int>::max();
  bpmDistance(x,y,error,ratio);
  if (fabs(error)<epsilon) return y;
  int a = int(x+0.5);
  int b = int(y+0.5);
  while (fabs(error) > epsilon) {
    bpmDistance(a,b,error,ratio);
    int remainder = a%b;
    a=b;
    b=remainder;
    //if(x<1) return 1;
  }
  return a;
}


inline
std::vector<Real> roundBpms(const std::vector<Real>& bpms) {
  Real epsilon = 3; // 3%
  Real mainBpm=bpms[0];
  std::vector<Real> harmonicBpms;
  harmonicBpms.reserve(bpms.size());
  for (int i=0; i<int(bpms.size()); i++) {
    Real ratio=bpms[0]/mainBpm;
    if (ratio < Real(1.0)) ratio = 1.0/ratio;
    ratio = round(ratio*10.)/10.; // rounding to 1 float pos
    int iRatio = int(ratio);
    if (ratio-iRatio <= 0.100001) { // allow 2.9, 3.1 be considered as 3
      harmonicBpms.push_back(bpms[i]);
    }
    else {
      if ((ratio-iRatio) == 0.5) { // only interested in pure halfs
        harmonicBpms.push_back(bpms[i]);
        harmonicBpms.push_back(greatestCommonDivisor(bpms[i], mainBpm,epsilon));
      }
    }
  }
  return harmonicBpms;
}


// original postprocessticks from essentia 1.0
inline
std::vector<Real> postProcessTicks(const std::vector<Real>& origticks) {
  if (origticks.size() < 3) return origticks;

  // find the most likely beat period
  const int nticks = origticks.size();
  std::vector<Real> dticks(nticks-1);

  for (int i=0; i<nticks-1; i++) dticks[i] = origticks[i+1] - origticks[i];

  // we might have had 6 secs frames during which we didn't find any beat, in which
  // case we'll have one huge dtick value, which we actually want to prune
  for (int i=0; i<(int)dticks.size(); i++) {
    if (dticks[i] > 2.) {
      dticks.erase(dticks.begin() + i);
      i--;
    }
  }

  const int nbins = 100;
  std::vector<int> dist(nbins);
  std::vector<Real> distx(nbins);

  hist(&dticks[0], nticks-1, &dist[0], &distx[0], nbins);

  int maxidx = max_element(dist.begin(), dist.end()) - dist.begin();
  Real maxbinCenter = distx[maxidx];

  // find the longest sequence of beats which has a fixed period of the previously
  // found value; use a tolerance of about 10%
  // Note: this favors high BPMs, because they will have more beats in the same amount of time
  int maxl = 0;
  int idx = 0;
  Real period = maxbinCenter;

  for (int startpos = 0; startpos < nticks-1; startpos++) {
    int l = longestChain(dticks, startpos, period, 0.1);
    if (l > maxl) {
      maxl = l;
      idx = startpos;
    }
  }

  if (idx == 0 && maxl == 0) {
    std::cout << "Internal error while processing the beats, returning the original ones" << std::endl;
    return origticks;
  }

  // let's assume those beats are correct, and try to replace all the other ones
  // with respect to the fixed period we have and the old positions of the beats
  std::deque<Real> ticks(origticks.begin() + idx,
                    origticks.begin() + idx + maxl + 1);

  // take this value as the period for the whole track
  Real targetPeriod = mean(dticks, idx, idx+maxl);
  // 0.15, because 0.1 might be too strict, while 0.2 will allow false positives more easily
  Real tolerance = 0.15 * targetPeriod;


  // do the beats after the current beat base
  Real cpos = ticks.back() + targetPeriod;
  std::deque<Real> remaining(origticks.begin() + idx + maxl + 1,
                        origticks.end());

  while (!remaining.empty()) {
    Real nbeat = remaining.front();

    if (nbeat < cpos - tolerance) {
      // too far before, drop next beat
      remaining.pop_front();
    }
    else {
      // right in our expected zone, adjust the estimated beat to the one
      // we actually found (NB: if we're too far away in front, just keep the
      // beat as is)
      if (nbeat < cpos + tolerance) {
        cpos = nbeat;
        remaining.pop_front();
      }

      // in any case, mark the beat and jump on the next one
      ticks.push_back(cpos);
      cpos += targetPeriod;
    }
  }

  // do the beats before the current beat base
  cpos = ticks.front() - targetPeriod;
  remaining = std::deque<Real>(origticks.begin(),
                          origticks.begin() + idx);

  while (!remaining.empty()) {
    Real nbeat = remaining.back();

    if (nbeat > cpos + tolerance) {
      // too far after, drop beat
      remaining.pop_back();
    }
    else {
      // right in our expected zone, adjust the estimated beat to the one
      // we actually found
      if (nbeat > cpos - tolerance) {
        cpos = nbeat;
        remaining.pop_back();
      }

      // in any case, mark the beat and jump on the next one
      ticks.push_front(cpos);
      cpos -= targetPeriod;
    }
  }


  return std::vector<Real>(ticks.begin(), ticks.end());
}

  // modified version of the postprocessticks from tempotapticks, so it does not
// tend to favour fast bpms
inline
std::vector<Real> postProcessTicks(const std::vector<Real>& origticks,
                                   const std::vector<Real>& ticksAmplitudes,
                                   const Real& preferredPeriod) {
  if (origticks.size() < 3) return origticks;

  // find the most likely beat period
  const int nticks = origticks.size();
  std::vector<Real> dticks(nticks-1);

  for (int i=0; i<nticks-1; i++) dticks[i] = origticks[i+1] - origticks[i];

  // we might have had 6 secs frames during which we didn't find any beat, in which
  // case we'll have one huge dtick value, which we actually want to prune
  for (int i=0; i<(int)dticks.size(); i++) {
    if (dticks[i] > 2.) {
      dticks.erase(dticks.begin() + i);
      i--;
    }
  }

  const int nbins = 100;
  std::vector<int> dist(nbins);
  std::vector<Real> distx(nbins);

  hist(&dticks[0], dticks.size(), &dist[0], &distx[0], nbins);

  int maxidx = max_element(dist.begin(), dist.end()) - dist.begin();
  Real maxbinCenter = distx[maxidx];

  // find the longest sequence of beats which has a fixed period of the previously
  // found value; use a tolerance of about 10%
  // Note: this favors high BPMs, because they will have more beats in the same amount of time
  int maxl = 0;
  int idx = 0;
  Real period = maxbinCenter;

  //std::cout << "period: " << period << std::endl;
  for (int startpos = 0; startpos < nticks-1; startpos++) {
    int l = longestChain(dticks, startpos, period, 0.05);
    if (l > maxl) {
      maxl = l;
      idx = startpos;
    }
  }

  Real targetPeriod = preferredPeriod;
  if (idx ==0 && maxl==0) {
    idx = argmax(ticksAmplitudes);
  }
  // take this value as the period for the whole track
  else targetPeriod = mean(dticks, idx, idx+maxl);

  targetPeriod = (targetPeriod+preferredPeriod)/2.0;

  // if the targetPeriod is differs too much from the preferred period we
  // search for the tick with max amplitude and take that as the reference tick
  if (targetPeriod < 0.95*preferredPeriod || targetPeriod > 1.05*preferredPeriod) {
    idx = idx + argmax(std::vector<Real>(ticksAmplitudes.begin()+idx, ticksAmplitudes.begin()+idx+maxl+1));
    maxl = 0;
    targetPeriod = preferredPeriod;
    //std::cout << "Targets differ too much!. New target period will be the preferred one " << targetPeriod << std::endl;
  }

  Real originalTargetPeriod = targetPeriod;

  // let's assume those beats are correct, and try to replace all the other ones
  // with respect to the fixed period we have and the old positions of the beats
  std::deque<Real> ticks(origticks.begin() + idx,
                    origticks.begin() + idx + maxl + 1);

  //fix tolerance at no more than 30ms:
  Real tolerance =0.03;

  if (targetPeriod < 0.05) {
     std::cout << "PostProcessTicks: target Period too short. Returning the original ticks" << std::endl;
     return origticks;
  }
  Real cummulatedPeriod = targetPeriod;
  int nAccumulations = 1;


  // do the beats after the current beat base
  Real cpos = ticks.back() + targetPeriod;
  std::deque<Real> remaining(origticks.begin() + idx + maxl + 1,
                        origticks.end());

  while (!remaining.empty()) {
    Real nbeat = remaining.front();

    if (nbeat < cpos - tolerance) {
      // too far before, drop next beat
      remaining.pop_front();
    }
    else {
      // right in our expected zone, adjust the estimated beat to the one
      // we actually found (NB: if we're too far away in front, just keep the
      // beat as is)
      if (nbeat < cpos + tolerance) {
        cummulatedPeriod +=  (nbeat - (cpos - targetPeriod));
        nAccumulations++;
        targetPeriod = cummulatedPeriod/nAccumulations;
        //std::cout << "new target Period: " << targetPeriod << " bpm: " << 60./targetPeriod << std::endl;
        //std::cout << " \tbeat at: " << nbeat << " belongs to [" << cpos-tolerance << ", " << cpos+tolerance <<"], cpos: " << cpos <<  std::endl;
        cpos = nbeat;
        remaining.pop_front();
      }

      // in any case, mark the beat and jump on the next one
      ticks.push_back(cpos);
      cpos += targetPeriod;
    }
  }

  // do the beats before the current beat base
  cpos = ticks.front() - targetPeriod;
  remaining = std::deque<Real>(origticks.begin(),
                          origticks.begin() + idx);

  targetPeriod = originalTargetPeriod;
  cummulatedPeriod = targetPeriod;
  nAccumulations = 1;

  while (!remaining.empty()) {
    Real nbeat = remaining.back();

    if (nbeat > cpos + tolerance) {
      // too far after, drop beat
      remaining.pop_back();
    }
    else {
      // right in our expected zone, adjust the estimated beat to the one
      // we actually found
      if (nbeat > cpos - tolerance) {
        cummulatedPeriod += ((cpos + targetPeriod)-nbeat);
        nAccumulations++;
        targetPeriod = cummulatedPeriod/nAccumulations;
        //std::cout << "new target Period: " << targetPeriod << " bpm: " << 60./targetPeriod << std::endl;
        //std::cout << " \tbeat at: " << nbeat << " belongs to [" << cpos-tolerance << ", " << cpos+tolerance <<"], cpos: " << cpos <<  std::endl;
        cpos = nbeat;
        remaining.pop_back();
      }

      // in any case, mark the beat and jump on the next one
      ticks.push_front(cpos);
      cpos -= targetPeriod;
    }
  }

  return std::vector<Real>(ticks.begin(), ticks.end());
}
} // namespace essentia

#endif
