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

#include "pitchcontours.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PitchContours::name = "PitchContours";
const char* PitchContours::version = "1.0";
const char* PitchContours::description = DOC("This algorithm tracks a set of predominant pitch contours from an audio signal. This algorithm is intended to receive its \"frequencies\" and \"magnitudes\" inputs from the PitchSalienceFunctionPeaks algorithm outputs aggregated over all frames in the sequence. The output is a vector of estimated melody pitch values.\n"
"\n"
"When input vectors differ in size, an exception is thrown. Input vectors must not contain negative salience values otherwise an exception is thrown. Avoiding erroneous peak duplicates (peaks of the same cent bin) is up to the user's own control and is highly recommended, but no exception will be thrown.\n"
"\n"
"Recommended processing chain: (see [1]): EqualLoudness -> frame slicing with sample rate = 44100, frame size = 2048, hop size = 128 -> Windowing with Hann, x4 zero padding -> Spectrum -> SpectralPeaks -> PitchSalienceFunction (10 cents bin resolution) -> PitchSalienceFunctionPeaks.\n"
"\n"
"References:\n"
"  [1] J. Salamon and E. Gómez, \"Melody extraction from polyphonic music\n"
"  signals using pitch contour characteristics,\" IEEE Transactions on Audio,\n"
"  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.\n");

void PitchContours::configure() {
  _binResolution = parameter("binResolution").toReal();
  _peakFrameThreshold = parameter("peakFrameThreshold").toReal();
  _peakDistributionThreshold = parameter("peakDistributionThreshold").toReal();
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = parameter("hopSize").toInt();

  _timeContinuityInFrames = (parameter("timeContinuity").toReal() / 1000.0) * _sampleRate / _hopSize;
  _minDurationInFrames = (parameter("minDuration").toReal() / 1000.0) * _sampleRate / _hopSize;
  // pitch continuity during 1 frame
  _pitchContinuityInBins = parameter("pitchContinuity").toReal() * 1000.0 * _hopSize / _sampleRate / _binResolution;

  _frameDuration = _hopSize / _sampleRate;
}

void PitchContours::compute() {
  const vector<vector<Real> >& peakBins = _peakBins.get();
  const vector<vector<Real> >& peakSaliences = _peakSaliences.get();

  vector<vector<Real> >& contoursBins = _contoursBins.get();
  vector<vector<Real> >& contoursSaliences =_contoursSaliences.get();
  vector<Real>& contoursStartTimes = _contoursStartTimes.get();
  Real& duration = _duration.get();

  // do sanity checks
  if (peakBins.size() != peakSaliences.size()) {
    throw EssentiaException("PitchContours: peakBins and peakSaliences input vectors must have the same size");
  }
  _numberFrames = peakBins.size();
  duration = _numberFrames * _frameDuration;

  if (!_numberFrames) {
    // no peaks -> empty pitch contours output
    contoursBins.clear();
    contoursSaliences.clear();
    contoursStartTimes.clear();
    return;
  }

  for (size_t i=0; i<_numberFrames; i++) {

    if (peakBins[i].size() != peakSaliences[i].size()) {
      throw EssentiaException("PitchContours: peakBins and peakSaliences input vectors must have the same size");
    }

    int numPeaks = peakBins[i].size();
    if (numPeaks==0) {
      continue;
    }

    for (int j=0; j<numPeaks; j++) {
      if (peakSaliences[i][j] < 0) {
        throw EssentiaException("PitchContours: salience peaks values input must be non-negative");
      }
    }

  }

  // compute pitch contours

  // per-frame filtering
  _salientPeaksBins.clear();
  _salientPeaksValues.clear();
  _nonSalientPeaksBins.clear();
  _nonSalientPeaksValues.clear();

  _salientPeaksBins.resize(_numberFrames);
  _salientPeaksValues.resize(_numberFrames);
  _nonSalientPeaksBins.resize(_numberFrames);
  _nonSalientPeaksValues.resize(_numberFrames);

  vector<pair<size_t, size_t> > salientInFrame;

  for (size_t i=0; i<_numberFrames; i++) {
    if (peakSaliences[i].size() == 0) { // avoiding that argmax will return 0 on empty vector
      continue;
    }
    Real frameMinSalienceThreshold = _peakFrameThreshold * peakSaliences[i][argmax(peakSaliences[i])];
    for (size_t j=0; j<peakBins[i].size(); j++) {
      if (peakSaliences[i][j] < frameMinSalienceThreshold) {
        _nonSalientPeaksBins[i].push_back(peakBins[i][j]);
        _nonSalientPeaksValues[i].push_back(peakSaliences[i][j]);
      }
      else {
        salientInFrame.push_back(make_pair(i,j));
      }
    }
  }

  // gather distribution statistics for overall peak filtering

  vector <Real> allPeakValues;
  for (size_t i=0; i<salientInFrame.size(); i++) {
    size_t ii = salientInFrame[i].first;
    size_t jj = salientInFrame[i].second;
    allPeakValues.push_back(peakSaliences[ii][jj]);
  }
  Real salienceMean = mean(allPeakValues);
  Real overallMeanSalienceThreshold = salienceMean - stddev(allPeakValues, salienceMean) * _peakDistributionThreshold;

  // distribution-based filtering
  for (size_t i=0; i<salientInFrame.size(); i++) {
    size_t ii = salientInFrame[i].first;
    size_t jj = salientInFrame[i].second;
    if (peakSaliences[ii][jj] < overallMeanSalienceThreshold) {
      _nonSalientPeaksBins[ii].push_back(peakBins[ii][jj]);
      _nonSalientPeaksValues[ii].push_back(peakSaliences[ii][jj]);
    }
    else {
      _salientPeaksBins[ii].push_back(peakBins[ii][jj]);
      _salientPeaksValues[ii].push_back(peakSaliences[ii][jj]);
    }
  }

  // peak streaming
  while(true) {
    size_t index;
    vector<Real> contourBins;
    vector<Real> contourSaliences;

    trackPitchContour(index, contourBins, contourSaliences);

    if(contourBins.size() > 0) {
      // Check if contour exceeds the allowed minimum length. This requirement is not documented in
      // the reference [1], but was reported in personal communication with the author.

      if (contourBins.size() >= _minDurationInFrames) {
        contoursStartTimes.push_back(Real(index) * _frameDuration);
        contoursBins.push_back(contourBins);
        contoursSaliences.push_back(contourSaliences);
      }
    }
    else {
      break;  // no new contour was found
    }
  }
}

int PitchContours::findNextPeak(vector<vector<Real> >& peaksBins, vector<Real>& contourBins, size_t i, bool backward) {
  // order = 1 to search forewards, = -1 to search backwards
  // i refers to a frame to search in for the next peak
  Real distance;
  int best_peak_j = -1;
  Real previousBin;
  Real bestPeakDistance = _pitchContinuityInBins;

  for (size_t j=0; j<peaksBins[i].size(); j++) {
    previousBin = backward ? contourBins.front() : contourBins.back();
    distance = abs(previousBin - peaksBins[i][j]);

    if (distance < bestPeakDistance) {
      best_peak_j = j;
      bestPeakDistance = distance;
    }
  }
  return best_peak_j;
}

void PitchContours::removePeak(vector<vector<Real> >& peaksBins, vector<vector<Real> >& peaksValues, size_t i, int j) {
  peaksBins[i].erase(peaksBins[i].begin() + j);
  peaksValues[i].erase(peaksValues[i].begin() + j);
}

void PitchContours::trackPitchContour(size_t& index, vector<Real>& contourBins, vector<Real>& contourSaliences) {
  // find the highest salient peak through all frames
  size_t max_i;
  int max_j;
  Real maxSalience = 0;

  for (size_t i=0; i<_numberFrames; i++) {
    if (_salientPeaksValues[i].size() > 0) {
      int j = argmax(_salientPeaksValues[i]);
      if (_salientPeaksValues[i][j] > maxSalience) {
        maxSalience = _salientPeaksValues[i][j];
        max_i = i;
        max_j = j;
      }
    }
  }
  if (maxSalience == 0) {
    // no salient peaks left in the set -> no new contours added
    return;
  }

  vector<pair<size_t,int> > removeNonSalientPeaks;

  // start new contour with this peak
  index = max_i; // the starting index of the contour
  contourBins.push_back(_salientPeaksBins[index][max_j]);
  contourSaliences.push_back(_salientPeaksValues[index][max_j]);
  // remove the peak from salient peaks
  removePeak(_salientPeaksBins, _salientPeaksValues, index, max_j);

  // track forwards in time
  int gap=0, best_peak_j;
  for (size_t i=index+1; i<_numberFrames; i++) {
    // find salient peaks in the next frame
    best_peak_j = findNextPeak(_salientPeaksBins, contourBins, i);
    if (best_peak_j >= 0) {
      // salient peak was found
      contourBins.push_back(_salientPeaksBins[i][best_peak_j]);
      contourSaliences.push_back(_salientPeaksValues[i][best_peak_j]);
      removePeak(_salientPeaksBins, _salientPeaksValues, i, best_peak_j);
      gap = 0;
    }
    else {
      // no peaks were found -> use non-salient ones
      // track using non-salient peaks for up to 100 ms by default
      if (gap+1 > _timeContinuityInFrames) {
        // this frame would already exceed the gap --> stop forward tracking
        break;
      }
      best_peak_j = findNextPeak(_nonSalientPeaksBins, contourBins, i);
      if (best_peak_j >= 0) {
        contourBins.push_back(_nonSalientPeaksBins[i][best_peak_j]);
        contourSaliences.push_back(_nonSalientPeaksValues[i][best_peak_j]);
        removeNonSalientPeaks.push_back(make_pair(i, best_peak_j));
        gap += 1;
      }
      else {
        break; // no salient nor non-salient peaks were found -> end of contour
      }
    }
  }
  // remove all included non-salient peaks from the tail of the contour,
  // as the contour should always finish with a salient peak
  for (int g=0; g<gap; g++) { // FIXME is using erase() faster?
    contourBins.pop_back();
    contourSaliences.pop_back();
  }

  // track backwards in time
  if (index == 0) {
    // we reached the starting frame

    // check if the contour exceeds the allowed minimum length
    if (contourBins.size() < _timeContinuityInFrames) {
      contourBins.clear();
      contourSaliences.clear();
    }
    return;
  }

  gap = 0;
  for (size_t i=index-1;;) {
    // find salient peaks in the previous frame
    best_peak_j = findNextPeak(_salientPeaksBins, contourBins, i, true);
    if (best_peak_j >= 0) {
      // salient peak was found, insert forward contourBins.insert(contourBins.begin(), _salientPeaksBins[i][best_peak_j]);
      contourBins.insert(contourBins.begin(), _salientPeaksBins[i][best_peak_j]);
      contourSaliences.insert(contourSaliences.begin(), _salientPeaksValues[i][best_peak_j]);
      removePeak(_salientPeaksBins, _salientPeaksValues, i, best_peak_j);
      index--;
      gap = 0;
    } else {
      // no salient peaks were found -> use non-salient ones
      if (gap+1 > _timeContinuityInFrames) {
        // this frame would already exceed the gap --> stop backward tracking
        break;
      }
      best_peak_j = findNextPeak(_nonSalientPeaksBins, contourBins, i, true);
      if (best_peak_j >= 0) {
        contourBins.insert(contourBins.begin(), _nonSalientPeaksBins[i][best_peak_j]);
        contourSaliences.insert(contourSaliences.begin(), _nonSalientPeaksValues[i][best_peak_j]);
        removeNonSalientPeaks.push_back(make_pair(i, best_peak_j));
        index--;
        gap += 1;
      }
      else {
        // no salient nor non-salient peaks were found -> end of contour
        break;
      }
    }

    // manual check of loop conditions, as size_t cannot be negative and, therefore, conditions inside "for" cannot be used
    if (i > 0) {
     i--;
    }
    else {
      break;
    }
  }
  // remove non-salient peaks for the beginning of the contour,
  // as the contour should start with a salient peak
  contourBins.erase(contourBins.begin(), contourBins.begin() + gap);
  contourSaliences.erase(contourSaliences.begin(), contourSaliences.begin() + gap);
  index += gap;

  // remove all employed non-salient peaks for the list of available peaks
  for(size_t r=0; r<removeNonSalientPeaks.size(); r++) {
    size_t i_p = removeNonSalientPeaks[r].first;
    if (i_p < index || i_p > index + contourBins.size()) {
      continue;
    }
    int j_p = removeNonSalientPeaks[r].second;
    removePeak(_nonSalientPeaksBins, _nonSalientPeaksValues, i_p, j_p);
  }
}

