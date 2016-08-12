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

#include "pitchcontoursmultimelody.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PitchContoursMultiMelody::name = "PitchContoursMultiMelody";
const char* PitchContoursMultiMelody::category = "Pitch";
const char* PitchContoursMultiMelody::description = DOC("This algorithm post-processes a set of pitch contours into a sequence of mutliple f0 values in Hz.\n"
"This algorithm is intended to receive its \"contoursBins\", \"contoursSaliences\", and \"contoursStartTimes\" inputs from the PitchContours algorithm. The \"duration\" input corresponds to the time duration of the input signal. The output is a vector of estimated pitch values\n"
"\n"
"When input vectors differ in size, or \"numberFrames\" is negative, an exception is thrown. Input vectors must not contain negative start indices nor negative bin and salience values otherwise an exception is thrown.\n"
"\n"
"References:\n"
"  [1] J. Salamon and E. Gómez, \"Melody extraction from polyphonic music\n"
"  signals using pitch contour characteristics,\" IEEE Transactions on Audio,\n"
"  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.\n");

void PitchContoursMultiMelody::configure() {
  // configurable parameters
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = parameter("hopSize").toInt();
  _referenceFrequency = parameter("referenceFrequency").toReal();
  _binResolution = parameter("binResolution").toReal();
  _filterIterations = parameter("filterIterations").toInt();
  _guessUnvoiced = parameter("guessUnvoiced").toBool();

  // minimum and maximum allowed cent bins for contours
  Real minFrequency = parameter("minFrequency").toReal();
  Real maxFrequency = parameter("maxFrequency").toReal();
  Real binsInOctave = 1200.0 / _binResolution;
  Real numberBins = floor(6000.0 / _binResolution) - 1;
  _minBin = max(0.0, floor(binsInOctave * log2(minFrequency/_referenceFrequency) + 0.5));
  _maxBin = min(0.0 + numberBins, floor(binsInOctave * log2(maxFrequency/_referenceFrequency) + 0.5));

  _frameDuration = _hopSize / _sampleRate;
  _outlierMaxDistance = (1200.0+50)/_binResolution; // a bit more than 1 octave
  _duplicateMaxDistance = _outlierMaxDistance;
  _duplicateMinDistance = (1200.0-50)/_binResolution;

  // 5-second moving average
  int averagerSize = floor(5 / _frameDuration);
  averagerSize = averagerSize % 2 == 0 ? averagerSize + 1 : averagerSize; // make the size odd
  _averagerShift = averagerSize / 2;

  // conversion to hertz
  _centToHertzBase = pow(2, _binResolution / 1200.0);
}

void PitchContoursMultiMelody::compute() {

  const vector<vector<Real> >& contoursBins = _contoursBins.get();
  const vector<vector<Real> >& contoursSaliences = _contoursSaliences.get();
  const vector<Real>& contoursStartTimes = _contoursStartTimes.get();
  const Real& duration = _duration.get();

  vector<vector<Real> >& pitch = _pitch.get();
    
  _numberFrames = (size_t) round(duration / _frameDuration);
  _numberContours = contoursBins.size();
    
  _contoursStartIndices.resize(_numberContours);
  _contoursEndIndices.resize(_numberContours);
  _contoursBinsMean.resize(_numberContours);
  _contoursSaliencesTotal.resize(_numberContours);
  _contoursSaliencesMean.resize(_numberContours);
  _contoursBinsStddev.resize(_numberContours);
    
  _contoursSelected.clear();
  _contoursIgnored.clear();
    
  // get contour salience and pitch statistics
  for (size_t i=0; i<_numberContours; i++) {
    _contoursBinsMean[i] = mean(contoursBins[i]);
    _contoursBinsStddev[i] = stddev(contoursBins[i], _contoursBinsMean[i]);
    _contoursSaliencesMean[i] = mean(contoursSaliences[i]);
  }
    
  for (size_t i=0; i<_numberContours; i++) {
    _contoursStartIndices[i] = (size_t) round(contoursStartTimes[i] / _frameDuration);
    _contoursEndIndices[i] = _contoursStartIndices[i] + contoursBins[i].size() - 1;
    _contoursSaliencesTotal[i] = accumulate(contoursSaliences[i].begin(), contoursSaliences[i].end(), 0.0);
    _contoursSelected.push_back(i);
  }
    
  _contoursSelectedInitially = _contoursSelected;
  _contoursIgnoredInitially = _contoursIgnored;


  // do sanity checks
  if (duration < 0) {
    throw EssentiaException("PitchContoursMultiMelody: specified duration of the input signal must be non-negative");
  }

  if (_numberContours != contoursSaliences.size() && _numberContours != contoursStartTimes.size()) {
    throw EssentiaException("PitchContoursMelody: contoursBins, contoursSaliences, and contoursStartTimes input vectors must have the same size");
  }

  pitch.resize(_numberFrames);

  // no frames -> empty pitch vector output
  if (!_numberFrames) {
    return;
  }

  for (size_t i=0; i<_numberContours; i++) {
    if (contoursBins[i].size() != contoursSaliences[i].size()) {
      throw EssentiaException("PitchContoursMelody: contoursBins and contoursSaliences input vectors must have the same size");
    }
    if (contoursStartTimes[i] < 0) {
      throw EssentiaException("PitchContoursMelody: contoursStartTimes input vector must contain non-negative values");
    }
    for (size_t j=0; j<contoursBins[i].size(); j++) {
      if (contoursBins[i][j] < 0) {
        throw EssentiaException("PitchContoursMelody: contour bin numbers must be non-negative");
      }
      if (contoursSaliences[i][j] < 0) {
        throw EssentiaException("PitchContoursMelody: contour pitch saliences must be non-negative");
      }
    }
  }

  // no contours -> zero pitch vector output
  if (contoursBins.empty()) {
      vector<Real> zero;
      zero.push_back(0.0);
      for (int i=0; i<(int)pitch.size(); i++){
          pitch.push_back(zero);
      }
    return;
  }
    
  // final melody selection: for each frame, select the peak
  // belonging to the contour with the highest total salience

    
  // remove octave duplicates
  detectContourDuplicates(contoursBins);
  removeContourDuplicates();
  detectContourDuplicates(contoursBins);
  removeContourDuplicates();
    
  Real centBin=0, hertz;

  for (size_t i=0; i<_numberFrames; i++) {
      vector<Real> pitchLocal;
      for (size_t j=0; j<_contoursSelected.size(); j++) {
          size_t jj = _contoursSelected[j];
          if (_contoursStartIndices[jj] <= i && _contoursEndIndices[jj] >= i) {
              // current frame belongs to this contour
              size_t shift = i - _contoursStartIndices[jj];
              centBin = contoursBins[jj][shift];
              hertz = _referenceFrequency * pow(_centToHertzBase, centBin);
              if (centBin>_minBin && centBin<_maxBin){
                  pitchLocal.push_back(hertz);
              }
          }
      }
      if (!pitchLocal.size()) {
          pitchLocal.push_back(0);
      }
      pitch[i]=pitchLocal;
  }
  
}


void PitchContoursMultiMelody::computeMelodyPitchMean(const vector<vector<Real> >& contoursBins) {

  /*
    Additional suggestion by Justin Salamon: implement a soft bias against the lowest frequencies:
    if f < 150Hz --> bias = f / 150Hz * 0.3
    In our evaluation, the results are ok without such a bias, when only using a hard threshold of
    80Hz for the minimum frequency allowed for salience peaks. Therefore the bias is not implemented.
  */

  vector<Real> melodyPitchMeanSmoothed;
  Real sumSalience;

  // compute melody pitch mean (weighted mean for all present contours) for each frame
  Real previous = 0.0;
  for (size_t i=0; i<_numberFrames; i++) {
    _melodyPitchMean[i]=0.0;
    sumSalience = 0.0;
    for (size_t j=0; j<_contoursSelected.size(); j++) {
      size_t jj = _contoursSelected[j];
      if (_contoursStartIndices[jj] <= i && _contoursEndIndices[jj] >= i) {
        // current frame belongs to this contour
        size_t shift = i - _contoursStartIndices[jj];
        _melodyPitchMean[i] += _contoursSaliencesTotal[jj] * contoursBins[jj][shift];
        sumSalience += _contoursSaliencesTotal[jj];
      }
    }
    if (sumSalience > 0) {
      _melodyPitchMean[i] /= sumSalience;
    } else {
      // no contour was found for current frame --> use value from previous bin
      _melodyPitchMean[i] = previous;
    }
    previous = _melodyPitchMean[i];
  }

  // replace zeros from the beginnig by the first non-zero value
  for (size_t i=0; i<_numberFrames; i++) {
    if (_melodyPitchMean[i] > 0) {
      for (size_t j=0; j<i; j++) {
        _melodyPitchMean[j] = _melodyPitchMean[i];
      }
      break;
    }
  }

  // run 5-second moving average filter to smooth melody pitch mean
  // we want to align filter output for symmetrical averaging,
  // and we want the filter to return values on the edges as the averager output computed at these positions
  // to avoid smoothing to zero

  _movingAverage->input("signal").set(_melodyPitchMean);
  _movingAverage->output("signal").set(melodyPitchMeanSmoothed);
  _movingAverage->reset();

  _melodyPitchMean.resize(_numberFrames + _averagerShift, _melodyPitchMean.back());
  _melodyPitchMean.insert(_melodyPitchMean.begin(), _averagerShift, _melodyPitchMean.front());
  _movingAverage->compute();
  _melodyPitchMean = vector<Real>(melodyPitchMeanSmoothed.begin() + 2*_averagerShift, melodyPitchMeanSmoothed.end());
}

void PitchContoursMultiMelody::detectContourDuplicates(const vector<vector<Real> >& contoursBins) {
  /*
    To compare contour trajectories we compute the distance between their pitch values on a per-frame basis for the
    region in which they overlap, and compute the mean over this region. If the mean distance is within 1200+-50 cents,
    the contours are considered octave duplicates.

    There is no requirement on the length of overlap region, according to [1] and personal communication with the
    author, but it can be introduced. However, algorithm already works well without such a requirement.
  */

  _duplicates.clear();  // re-initialize

  for(size_t i=0; i<_contoursSelected.size(); i++) {
    size_t ii = _contoursSelected[i];

    for (size_t j=i+1; j<_contoursSelected.size(); j++) {
      size_t jj = _contoursSelected[j];
      size_t start, end;
      bool overlap = false;

      if (_contoursStartIndices[ii] >= _contoursStartIndices[jj]
          && _contoursStartIndices[ii] <= _contoursEndIndices[jj]) {
        // .......[CONTOUR1]......
        // ....[CONTOUR2].........
        // or
        // .......[CONTOUR1]......
        // ....[CONTOUR2.....]....
        start = _contoursStartIndices[ii];
        end = min(_contoursEndIndices[ii], _contoursEndIndices[jj]);
        overlap = true;
      }
      else if (_contoursStartIndices[jj] <= _contoursEndIndices[ii]
          && _contoursStartIndices[jj] >= _contoursStartIndices[ii]) {
        // ....[CONTOUR1].........
        // .......[CONTOUR2]......
        // or
        // ....[CONTOUR1.....]....
        // .......[CONTOUR2]......
        start = _contoursStartIndices[jj];
        end = min(_contoursEndIndices[ii], _contoursEndIndices[jj]);
        overlap = true;
      }
      if (overlap) {
        // compute the mean distance for overlap region
        Real distance = 0;
        size_t shift_i = start - _contoursStartIndices[ii];
        size_t shift_j = start - _contoursStartIndices[jj];

        for (size_t ioverlap=start; ioverlap<=end; ioverlap++) {
           distance += contoursBins[ii][shift_i] - contoursBins[jj][shift_j];
           shift_i++;
           shift_j++;
        }
        distance = abs(distance) / (end-start+ 1);
        // recode cents to bins
        if (distance > _duplicateMinDistance && distance < _duplicateMaxDistance) {
          // contours ii and jj differ for around 1200 cents (i.e., 1 octave) --> they are duplicates
          _duplicates.push_back(make_pair(ii,jj));
        }
      }
    }
  }
}


void PitchContoursMultiMelody::removeContourDuplicates() {

  // for each duplicates pair, remove the contour furtherst from melody pitch mean
  for (size_t c=0; c<_duplicates.size(); c++) {
    size_t ii = _duplicates[c].first;
    size_t jj = _duplicates[c].second;
      
      if (_contoursBinsMean[ii]<_minBin) {
          _contoursSelected.erase(std::remove(_contoursSelected.begin(), _contoursSelected.end(), ii), _contoursSelected.end());
          continue;
      }
      if (_contoursBinsMean[jj]<_minBin){
          _contoursSelected.erase(std::remove(_contoursSelected.begin(), _contoursSelected.end(), jj), _contoursSelected.end());
          continue;
      }
      if (_contoursBinsMean[ii]>_maxBin) {
          _contoursSelected.erase(std::remove(_contoursSelected.begin(), _contoursSelected.end(), ii), _contoursSelected.end());
          continue;
      }
      if (_contoursBinsMean[jj]>_maxBin){
          _contoursSelected.erase(std::remove(_contoursSelected.begin(), _contoursSelected.end(), jj), _contoursSelected.end());
          continue;
      }
    
      if (_contoursSaliencesMean[ii]>=_contoursSaliencesMean[jj]){
          _contoursSelected.erase(std::remove(_contoursSelected.begin(), _contoursSelected.end(), jj), _contoursSelected.end());
      }else{
         _contoursSelected.erase(std::remove(_contoursSelected.begin(), _contoursSelected.end(), ii), _contoursSelected.end());
      }
        
  }
}

void PitchContoursMultiMelody::removePitchOutliers() {

  // compute average melody pitch mean on the intervals corresponding to all contour
  // remove pitch outliers by deleting contours at a distance more that one octave from melody pitch mean

  for (std::vector<size_t>::iterator iter = _contoursSelected.begin(); iter != _contoursSelected.end();) {
    size_t ii = *iter;
    Real contourMelodyPitchMean = accumulate(_melodyPitchMean.begin() + _contoursStartIndices[ii], _melodyPitchMean.begin() + _contoursEndIndices[ii] + 1, 0.0);
    contourMelodyPitchMean /= (_contoursEndIndices[ii] - _contoursStartIndices[ii] + 1);
    if (abs(_contoursBinsMean[ii] - contourMelodyPitchMean) > _outlierMaxDistance) {
      // remove contour
      iter = _contoursSelected.erase(iter);
      if (_guessUnvoiced) {
        _contoursIgnored.push_back(ii);
      }
    }
    else {
      ++iter;
    }
  }
}

