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

#include "pitchcontoursmelody.h"
#include "essentiamath.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PitchContoursMelody::name = "PitchContoursMelody";
const char* PitchContoursMelody::version = "1.0";
const char* PitchContoursMelody::description = DOC("This algorithm converts a set of pitch contours into a sequence of predominant f0 values in Hz by taking the value of the most predominant contour in each frame.\n"
"This algorithm is intended to receive its \"contoursBins\", \"contoursSaliences\", and \"contoursStartTimes\" inputs from the PitchContours algorithm. The \"duration\" input corresponds to the time duration of the input signal. The output is a vector of estimated pitch values and a vector of confidence values.\n"
"\n"
"Note that \"pitchConfidence\" can be negative in the case of \"guessUnvoiced\"=True: the absolute values represent the confidence, negative values correspond to segments for which non-salient contours where selected, zero values correspond to non-voiced segments.\n"
"\n"
"When input vectors differ in size, or \"numberFrames\" is negative, an exception is thrown. Input vectors must not contain negative start indices nor negative bin and salience values otherwise an exception is thrown.\n"
"\n"
"Recommended processing chain: (see [1]): EqualLoudness -> frame slicing with sample rate = 44100, frame size = 2048, hop size = 128 -> Windowing with Hann, x4 zero padding -> Spectrum -> SpectralPeaks -> PitchSalienceFunction -> PitchSalienceFunctionPeaks -> PitchContours.\n"
"\n"
"References:\n"
"  [1] J. Salamon and E. Gómez, \"Melody extraction from polyphonic music\n"
"  signals using pitch contour characteristics,\" IEEE Transactions on Audio,\n"
"  Speech, and Language Processing, vol. 20, no. 6, pp. 1759–1770, 2012.\n");

void PitchContoursMelody::configure() {
  // configurable parameters
  _voicingTolerance = parameter("voicingTolerance").toReal();
  _sampleRate = parameter("sampleRate").toReal();
  _hopSize = parameter("hopSize").toInt();
  _referenceFrequency = parameter("referenceFrequency").toReal();
  _binResolution = parameter("binResolution").toReal();
  _filterIterations = parameter("filterIterations").toInt();
  _voiceVibrato = parameter("voiceVibrato").toBool();
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

  _vibratoPitchStddev = 40 / _binResolution; // 40 cents

  // parameters voice vibrato detection
  // frame size computed given than we need 350ms of audio (as with default settings)
  Real _vibratoSampleRate = _sampleRate / _hopSize;
  _vibratoFrameSize = int(0.350 * _vibratoSampleRate);
  _vibratoHopSize = 1;
  _vibratoZeroPaddingFactor = 4;
  _vibratoFFTSize = _vibratoFrameSize * _vibratoZeroPaddingFactor;
  _vibratoFFTSize = pow(2, ceil(log(_vibratoFFTSize)/log(2)));
  _vibratoMinFrequency = 5.0;
  _vibratoMaxFrequency = 8.0;
  _vibratodBDropLobe = 15;
  _vibratodBDropSecondPeak = 20;

  // conversion to hertz
  _centToHertzBase = pow(2, _binResolution / 1200.0);

  // configure algorithms
  _movingAverage->configure("size", averagerSize);
  _frameCutter->configure("frameSize", _vibratoFrameSize, "hopSize", _vibratoHopSize, "startFromZero", true);
  _spectrum->configure("size", _vibratoFFTSize);
  _windowing->configure("type", "hann");
  _windowing->configure("zeroPadding", _vibratoFFTSize - _vibratoFrameSize);
  _spectralPeaks->configure("sampleRate", _vibratoSampleRate);
  _spectralPeaks->configure("maxPeaks", 3); // we are only interested in the three most prominent peaks
  _spectralPeaks->configure("orderBy", "magnitude");
}

void PitchContoursMelody::compute() {

  const vector<vector<Real> >& contoursBins = _contoursBins.get();
  const vector<vector<Real> >& contoursSaliences = _contoursSaliences.get();
  const vector<Real>& contoursStartTimes = _contoursStartTimes.get();
  const Real& duration = _duration.get();

  vector <Real>& pitch = _pitch.get();
  vector <Real>& pitchConfidence = _pitchConfidence.get();

  // do sanity checks

  if (duration < 0) {
    throw EssentiaException("PitchContoursMelody: specified duration of the input signal must be non-negative");
  }

  _numberFrames = (size_t) round(duration / _frameDuration);
  _numberContours = contoursBins.size();

  if (_numberContours != contoursSaliences.size() && _numberContours != contoursStartTimes.size()) {
    throw EssentiaException("PitchContoursMelody: contoursBins, contoursSaliences, and contoursStartTimes input vectors must have the same size");
  }

  pitch.resize(_numberFrames);
  pitchConfidence.resize(_numberFrames);

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
    fill(pitch.begin(), pitch.end(), (Real) 0.0);
    fill(pitchConfidence.begin(), pitchConfidence.end(), (Real) 0.0);
    return;
  }

  // voicing detection
  voicingDetection(contoursBins, contoursSaliences, contoursStartTimes);

  // create a list of all possible duplicates
  detectContourDuplicates(contoursBins);

  // filter octave errors and pitch outliers
  _melodyPitchMean.resize(_numberFrames);

  for (int i=0; i<_filterIterations; i++) {
    computeMelodyPitchMean(contoursBins);
    removeContourDuplicates();
    computeMelodyPitchMean(contoursBins);
    removePitchOutliers();
  }

  // final melody selection: for each frame, select the peak
  // belonging to the contour with the highest total salience

  Real centBin=0, hertz;

  for (size_t i=0; i<_numberFrames; i++) {
    Real maxSalience = 0;
    Real confidence = 0;
    for (size_t j=0; j<_contoursSelected.size(); j++) {
      size_t jj = _contoursSelected[j];
      if (_contoursStartIndices[jj] <= i && _contoursEndIndices[jj] >= i) {
        // current frame belongs to this contour
        size_t shift = i - _contoursStartIndices[jj];
        if (_contoursSaliencesTotal[jj] > maxSalience) {
          maxSalience = _contoursSaliencesTotal[jj];
          confidence = _contoursSaliencesMean[jj];
          centBin = contoursBins[jj][shift];
        }
      }
    }

    if (maxSalience==0 && _guessUnvoiced) {
      for (size_t j=0; j<_contoursIgnored.size(); j++) {
        size_t jj = _contoursIgnored[j];
        if (_contoursStartIndices[jj] <= i && _contoursEndIndices[jj] >= i) {
          // current frame belongs to this contour
          size_t shift = i - _contoursStartIndices[jj];
          if (_contoursSaliencesTotal[jj] > maxSalience) {
            maxSalience = _contoursSaliencesTotal[jj]; // store salience with negative sign in the case of unvoiced frames
            confidence = 0.0 - _contoursSaliencesMean[jj];
            centBin = contoursBins[jj][shift];
          }
        }
      }
    }

    if(maxSalience != 0) {
      // a peak was found, convert cent bins to Hertz
      // slow formula: _referenceFrequency * pow(2, centBin*_binResolution / 1200.0);
      hertz = _referenceFrequency * pow(_centToHertzBase, centBin);
    } else {
      hertz = 0;
    }
    pitch[i] = hertz;
    pitchConfidence[i] = confidence;
  }
}

bool PitchContoursMelody::detectVoiceVibrato(vector<Real> contourBins, const Real binMean) {

  /*
    Algorithm details are taken from personal communication with Justin Salamon. There should be only one (and it should be
    the highest) peak between 5 and 8 Hz, associated with human voice vibrato.  If there is more than 1 peak in this interval,
    we may not be sure in vibrato --> go to search in next frame.

    Find the 2nd and the 3rd highest peaks above 8Hz (we don't care in peaks below 5Hz, and they are normally not expected
    to appear). The second peak should be 15 dBs quieter, and the third peak should be 20 dBs quieter than the highest peak.
    If so, the voice peak is prominent enough --> human voice vibrato found in the contour.
  */

  if (!_voiceVibrato) {
    return false;
  }

  // subtract mean from the contour pitch trajectory
  for (size_t i=0; i<contourBins.size(); i++) {
    contourBins[i] -= binMean;
  }

  // apply FFT and check for a prominent peak in the expected frequency range for human vibrato (5-8Hz)

  vector<Real> frame;
  _frameCutter->input("signal").set(contourBins);
  _frameCutter->output("frame").set(frame);

  vector<Real> frameWindow;
  _windowing->input("frame").set(frame);
  _windowing->output("frame").set(frameWindow);

  vector<Real> vibratoSpectrum;
  _spectrum->input("frame").set(frameWindow);
  _spectrum->output("spectrum").set(vibratoSpectrum);

  vector<Real> peakFrequencies;
  vector<Real> peakMagnitudes;
  _spectralPeaks->input("spectrum").set(vibratoSpectrum);
  _spectralPeaks->output("frequencies").set(peakFrequencies);
  _spectralPeaks->output("magnitudes").set(peakMagnitudes);

  _frameCutter->reset();

  while (true) {
    // get a frame
    _frameCutter->compute();
    if (!frame.size()) {
      break;
    }

    _windowing->compute();
    _spectrum->compute();
    _spectralPeaks->compute();

    int numberPeaks = peakFrequencies.size();
    if (!numberPeaks) {
      continue;
    }

    if (peakFrequencies[0] < _vibratoMinFrequency || peakFrequencies[0] > _vibratoMaxFrequency) {
      continue;
    }

    if (numberPeaks > 1) {  // there is at least one extra peak
      if (peakFrequencies[1] <= _vibratoMaxFrequency) {
        continue;
      }
      if (20 * log10(peakMagnitudes[0]/peakMagnitudes[1]) < _vibratodBDropLobe) {
        continue;
      }
    }

    if (numberPeaks > 2) {  // there is a second extra peak
      if (peakFrequencies[2] <= _vibratoMaxFrequency) {
        continue;
      }
      if (20 * log10(peakMagnitudes[0]/peakMagnitudes[2]) < _vibratodBDropSecondPeak) {
        continue;
      }
    }
    // prominent peak associated with voice is found
    return true;
  }
  return false;
}

void PitchContoursMelody::voicingDetection(const vector<vector<Real> >& contoursBins,
                                           const vector<vector<Real> >& contoursSaliences,
                                           const vector<Real>& contoursStartTimes) {

  _contoursStartIndices.resize(_numberContours);
  _contoursEndIndices.resize(_numberContours);
  _contoursBinsMean.resize(_numberContours);
  _contoursSaliencesTotal.resize(_numberContours);
  _contoursSaliencesMean.resize(_numberContours);
  _contoursBinsStddev.resize(_numberContours);  // TODO make a local variable

  _contoursSelected.clear();
  _contoursIgnored.clear();

  vector<Real> contoursBinsMin;
  vector<Real> contoursBinsMax;
  contoursBinsMin.resize(_numberContours);
  contoursBinsMax.resize(_numberContours);

  // get contour salience and pitch statistics
  for (size_t i=0; i<_numberContours; i++) {
    _contoursBinsMean[i] = mean(contoursBins[i]);
    _contoursBinsStddev[i] = stddev(contoursBins[i], _contoursBinsMean[i]);
    _contoursSaliencesMean[i] = mean(contoursSaliences[i]);
    contoursBinsMin[i] = contoursBins[i][argmin(contoursBins[i])];
    contoursBinsMax[i] = contoursBins[i][argmax(contoursBins[i])];
  }
  Real averageSalienceMean = mean(_contoursSaliencesMean);
  Real salienceThreshold = averageSalienceMean - _voicingTolerance * stddev(_contoursSaliencesMean, averageSalienceMean);

  // voicing detection
  for (size_t i=0; i<_numberContours; i++) {
    // ignore contours with peaks outside of the allowed range
    if (contoursBinsMin[i] >= _minBin && contoursBinsMax[i] <= _maxBin) {
      if (_contoursSaliencesMean[i] >= salienceThreshold || _contoursBinsStddev[i] > _vibratoPitchStddev
                                            || detectVoiceVibrato(contoursBins[i], _contoursBinsMean[i]))  {
        _contoursStartIndices[i] = (size_t) round(contoursStartTimes[i] / _frameDuration);
        _contoursEndIndices[i] = _contoursStartIndices[i] + contoursBins[i].size() - 1;
        _contoursSaliencesTotal[i] = accumulate(contoursSaliences[i].begin(), contoursSaliences[i].end(), 0.0);
        _contoursSelected.push_back(i);
      }
      else {
        if (_guessUnvoiced) {
          _contoursStartIndices[i] = (size_t) round(contoursStartTimes[i] / _frameDuration);
          _contoursEndIndices[i] = _contoursStartIndices[i] + contoursBins[i].size() - 1;
          _contoursSaliencesTotal[i] = accumulate(contoursSaliences[i].begin(), contoursSaliences[i].end(), 0.0);
          _contoursIgnored.push_back(i);
        }
      }
    }
  }
  _contoursSelectedInitially = _contoursSelected;
  _contoursIgnoredInitially = _contoursIgnored;
}

void PitchContoursMelody::computeMelodyPitchMean(const vector<vector<Real> >& contoursBins) {

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

void PitchContoursMelody::detectContourDuplicates(const vector<vector<Real> >& contoursBins) {
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


void PitchContoursMelody::removeContourDuplicates() {

  // each iteration we start with all contours that passed the voiding detection stage,
  // but use the most recently computed melody pitch mean.

  // reinitialize the list of selected contours
  _contoursSelected = _contoursSelectedInitially;
  _contoursIgnored = _contoursIgnoredInitially;

  // compute average melody pitch mean on the intervals corresponding to all contours
  vector<Real> contoursMelodyPitchMean;
  contoursMelodyPitchMean.resize(_numberContours);
  for (size_t i=0; i<_contoursSelected.size(); i++) {
    size_t ii = _contoursSelected[i];
    contoursMelodyPitchMean[ii] = accumulate(_melodyPitchMean.begin() + _contoursStartIndices[ii], _melodyPitchMean.begin() + _contoursEndIndices[ii] + 1, 0);
    contoursMelodyPitchMean[ii] /= (_contoursEndIndices[ii] - _contoursStartIndices[ii] + 1);
  }

  // for each duplicates pair, remove the contour furtherst from melody pitch mean
  for (size_t c=0; c<_duplicates.size(); c++) {
    size_t ii = _duplicates[c].first;
    size_t jj = _duplicates[c].second;
    Real ii_distance = abs(_contoursBinsMean[ii] - contoursMelodyPitchMean[ii]);
    Real jj_distance = abs(_contoursBinsMean[jj] - contoursMelodyPitchMean[jj]);
    if (ii_distance < jj_distance) {
      // remove contour jj
      _contoursSelected.erase(std::remove(_contoursSelected.begin(), _contoursSelected.end(), jj), _contoursSelected.end());
      if (_guessUnvoiced) {
        _contoursIgnored.push_back(jj);
      }
    } else {
      // remove contour ii
      _contoursSelected.erase(std::remove(_contoursSelected.begin(), _contoursSelected.end(), ii), _contoursSelected.end());
      if (_guessUnvoiced) {
        _contoursIgnored.push_back(ii);
      }
    }
  }
}

void PitchContoursMelody::removePitchOutliers() {

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

