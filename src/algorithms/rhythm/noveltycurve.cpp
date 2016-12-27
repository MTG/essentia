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

#include "algorithmfactory.h"
#include "essentiamath.h"
#include "noveltycurve.h"

using namespace std;

namespace essentia {
namespace standard {

const char* NoveltyCurve::name = "NoveltyCurve";
const char* NoveltyCurve::category = "Rhythm";
const char* NoveltyCurve::description = DOC("This algorithm computes the \"novelty curve\" (Grosche & Müller, 2009) onset detection function. The algorithm expects as an input a frame-wise sequence of frequency-bands energies or spectrum magnitudes as originally proposed in [1] (see FrequencyBands and Spectrum algorithms). Novelty in each band (or frequency bin) is computed as a derivative between log-compressed energy (magnitude) values in consequent frames. The overall novelty value is then computed as a weighted sum that can be configured using 'weightCurve' parameter. The resulting novelty curve can be used for beat tracking and onset detection (see BpmHistogram and Onsets).\n"
"\n"
"Notes:\n"
"- Recommended frame/hop size for spectrum computation is 2048/1024 samples (44.1 kHz sampling rate) [2].\n"
"- Log compression is applied with C=1000 as in [1].\n"
"- Frequency bands energies (see FrequencyBands) as well as bin magnitudes for the whole spectrum can be used as an input. The implementation for the original algorithm [2] works with spectrum bin magnitudes for which novelty functions are computed separately and are then summarized into bands.\n"
"- In the case if 'weightCurve' is set to 'hybrid' a complex combination of flat, quadratic, linear and inverse quadratic weight curves is used. It was reported to improve performance of beat tracking in some informal in-house experiments (Note: this information is probably outdated).\n"
"\n"
"References:\n"
"  [1] P. Grosche and M. Müller, \"A mid-level representation for capturing\n"
"  dominant tempo and pulse information in music recordings,\" in\n"
"  International Society for Music Information Retrieval Conference\n"
"  (ISMIR’09), 2009, pp. 189–194.\n"
"  [2] Tempogram Toolbox (Matlab implementation),\n"
"  http://resources.mpi-inf.mpg.de/MIR/tempogramtoolbox\n");


vector<Real> NoveltyCurve::weightCurve(int size, WeightType type) {
  vector<Real> result(size, 0.0);
  int halfSize = size/2;
  int sqrHalfSize = halfSize*halfSize;
  int sqrSize = size*size;

  // NB:some of these curves have a +1 so we don't reach zero!!
  switch(type) {
    case FLAT:
      fill(result.begin(), result.end(), Real(1.0));
      break;
    case TRIANGLE:
      for (int i=0; i<halfSize; i++) {
        result[i] = result[size-1-i] = i+1;
      }
      if ((size&1) == 1) result[halfSize] = size/2; // for odd sizes
      break;
    case INVERSE_TRIANGLE:
      for (int i=0; i<halfSize; i++) {
        result[i] = result[size-1-i] = halfSize-i;
      }
      break;
    case PARABOLA:
      for (int i=0; i<halfSize; i++) {
        result[i] = result[size-1-i] = (halfSize-i)*(halfSize-i);
      }
      break;
    case INVERSE_PARABOLA:
      for (int i=0; i<halfSize; i++) {
        result[i] = sqrHalfSize - (halfSize-i)*(halfSize-i)+1;
        result[size-1-i] =  result[i];
      }
      if ((size&1) == 1) result[halfSize] = halfSize; // for odd sizes
      break;
    case LINEAR:
      for (int i=0; i<size; i++) result[i] = i+1;
      break;
    case QUADRATIC:
      for (int i=0; i<size; i++) result[i] = i*i+1;
      break;
    case INVERSE_QUADRATIC:
      for (int i=0; i<size; i++) result[i] = sqrSize - i*i;
      break;
    case SUPPLIED:
      result = parameter("weightCurve").toVectorReal();
      if (int(result.size()) != size) {
        throw EssentiaException("NoveltyCurve::weightCurve, the size of the supplied weights must be the same as the number of the frequency bands", size);
      }
      break;
    default:
      throw EssentiaException("Weighting Curve type not known");
    }

  //Real max = *max_element(result.begin(), result.end());
  //if (max == 0) throw EssentiaException("Weighting curves has null maximum");
  //for (int i=0; i<size; i++) result[i] /= max;

  return result;
}

/**
 * Compute the novelty curve for a single variable (energy band, spectrum bin, ...).
 * Resulting output vector size is equal to the input vector. The first value is always set 
 * to 0 because for it the derivative cannot be defined.
 */
vector<Real> NoveltyCurve::noveltyFunction(const vector<Real>& spec, Real C, int meanSize) {
  int size = spec.size();
  int dsize = size - 1;

  vector<Real> logSpec(size, 0.0), novelty(dsize, 0.0);
  for (int i=0; i<size; i++) logSpec[i] = log10(1 + C*spec[i]);

  // differentiate log spec and keep only positive variations
  for (int i=1; i<size; i++) {
    Real d = logSpec[i] - logSpec[i-1];
    if (d>0) novelty[i-1] = d;
  }

  // subtract local mean
  for (int i=0; i<dsize; i++) {
    int start = i - meanSize/2, end = i + meanSize/2;
    // TODO: decide on which option to choose

    // Nico adjust
    //start = max(start, 0);
    //end = min(end, size-1);
    
    // Edu adjust
    if (start<0 && end>=dsize) {start=0; end=dsize;}
    else {
      if (start<0) { start=0; end=meanSize;}
      if (end>=dsize) { end=dsize; start=dsize-meanSize;}
    }
  
    Real m = essentia::mean(novelty, start, end);
    if (novelty[i] < m) novelty[i]=0.0;
    else novelty[i] -= m;
  }
  if (_normalize) {
    Real maxValue = *max_element(novelty.begin(), novelty.end());
    if (maxValue != 0) {
      vector<Real>::iterator it = novelty.begin();
      for (;it!=novelty.end(); ++it) *it /= maxValue;
    }
  }
  Algorithm * mavg = AlgorithmFactory::create("MovingAverage", "size", meanSize);
  vector<Real> novelty_ma;
  mavg->input("signal").set(novelty);
  mavg->output("signal").set(novelty_ma);
  mavg->compute();
  delete mavg;
  return novelty_ma;
}

void NoveltyCurve::configure() {
  string type = parameter("weightCurveType").toString();
  if (type == "flat") _type = FLAT;
  else if (type == "triangle") _type = TRIANGLE;
  else if (type == "inverse_triangle") _type = INVERSE_TRIANGLE;
  else if (type == "parabola") _type = PARABOLA;
  else if (type == "inverse_parabola") _type = INVERSE_PARABOLA;
  else if (type == "linear") _type = LINEAR;
  else if (type == "quadratic") _type = QUADRATIC;
  else if (type == "inverse_quadratic") _type = INVERSE_QUADRATIC;
  else if (type == "supplied") _type = SUPPLIED;
  else if (type == "hybrid") _type = HYBRID;
  _frameRate = parameter("frameRate").toReal();
  _normalize = parameter("normalize").toBool();
}


void NoveltyCurve::compute() {
  const vector<vector<Real> >& frequencyBands = _frequencyBands.get();
  vector<Real>& novelty = _novelty.get();
  if (frequencyBands.empty())
    throw EssentiaException("NoveltyCurve::compute, cannot compute from an empty input matrix");

  int nFrames = frequencyBands.size();
  int nBands = (int)frequencyBands[0].size();
  //vector<Real> weights = weightCurve(nBands);
  novelty.resize(nFrames-1);
  fill(novelty.begin(), novelty.end(), Real(0.0));

  vector<vector<Real> > t_frequencyBands = essentia::transpose(frequencyBands); // [bands x frames]
  vector<vector<Real> > noveltyBands(nBands);

  int meanSize = int(0.1 * _frameRate); // integral number of frames in 2*0.05 second

  // compute novelty for each sub-band
  meanSize += (meanSize % 2); // force even size // TODO: why?
  for (int bandIdx=0; bandIdx<nBands; bandIdx++) {
    noveltyBands[bandIdx] = noveltyFunction(t_frequencyBands[bandIdx], 1000, meanSize);
  }


  //sum novelty on all bands (weighted) to get a single novelty value per frame
  noveltyBands = essentia::transpose(noveltyBands); // back to [frames x bands]

  // TODO: weight curves should be pre-computed in configure() method
  if (_type == HYBRID) {
    // EAylon: By trial-&-error I found that combining weightings (flat, quadratic,
    // linear and inverse quadratic) was giving better results.   
    vector<Real> aweights = weightCurve(nBands, FLAT);
    vector<Real> bweights = weightCurve(nBands, QUADRATIC);
    vector<Real> cweights = weightCurve(nBands, LINEAR);
    vector<Real> dweights = weightCurve(nBands, INVERSE_QUADRATIC);

    vector<Real> bnovelty(nFrames-1, 0.0);
    vector<Real> cnovelty(nFrames-1, 0.0);
    vector<Real> dnovelty(nFrames-1, 0.0);

    for (int frameIdx=0; frameIdx<nFrames-1; frameIdx++) { // noveltyBands is a derivative whose size is nframes-1
      for (int bandIdx=0; bandIdx<nBands; bandIdx++) {
        novelty[frameIdx] += aweights[bandIdx] * noveltyBands[frameIdx][bandIdx];
        bnovelty[frameIdx] += bweights[bandIdx] * noveltyBands[frameIdx][bandIdx];
        cnovelty[frameIdx] += cweights[bandIdx] * noveltyBands[frameIdx][bandIdx];
        dnovelty[frameIdx] += dweights[bandIdx] * noveltyBands[frameIdx][bandIdx];
      }
    }
    for (int frameIdx=0; frameIdx<nFrames-1; frameIdx++) {
      // TODO why multiplication instead of sum (or mean)? 
      novelty[frameIdx] *= bnovelty[frameIdx];
      novelty[frameIdx] *= cnovelty[frameIdx];
      novelty[frameIdx] *= dnovelty[frameIdx];
    }
  }
  else {
    // TODO weight curve should be pre-computed in configure() method
    vector<Real> weights = weightCurve(nBands, _type);

    for (int frameIdx=0; frameIdx<nFrames-1; frameIdx++) {
      for (int bandIdx=0; bandIdx<nBands; bandIdx++) {
        novelty[frameIdx] += weights[bandIdx] * noveltyBands[frameIdx][bandIdx];
      }
    }
  }

  // smoothing
  Algorithm * mavg = AlgorithmFactory::create("MovingAverage", "size", meanSize);
  vector<Real> novelty_ma;
  mavg->input("signal").set(novelty);
  mavg->output("signal").set(novelty_ma);
  mavg->compute();
  delete mavg;
  novelty.assign(novelty_ma.begin(), novelty_ma.end());
}

void NoveltyCurve::reset() {
  Algorithm::reset();
}

} // namespace standard
} // namespace essentia


#include "poolstorage.h"
#include "algorithmfactory.h"

namespace essentia {
namespace streaming {

const char* NoveltyCurve::name = standard::NoveltyCurve::name;
const char* NoveltyCurve::description = standard::NoveltyCurve::description;

NoveltyCurve::NoveltyCurve() : AlgorithmComposite() {

  _noveltyCurve = standard::AlgorithmFactory::create("NoveltyCurve");
  _poolStorage = new PoolStorage<vector<Real> >(&_pool, "internal.frequencyBands");

  declareInput(_frequencyBands, 1, "frequencyBands", "the frequency bands");
  declareOutput(_novelty, 0, "novelty", "the novelty curve as a single vector");

  _frequencyBands >> _poolStorage->input("data"); // attach input proxy

  // Need to set the buffer type to multiple frames as all the values 
  // are output all at once
  _novelty.setBufferType(BufferUsage::forMultipleFrames);
}


NoveltyCurve::~NoveltyCurve() {
  delete _noveltyCurve;
  delete _poolStorage;
}


void NoveltyCurve::reset() {
  AlgorithmComposite::reset();
  _noveltyCurve->reset();
}


AlgorithmStatus NoveltyCurve::process() {
  if (!shouldStop()) return PASS;

  vector<Real> novelty;
  _noveltyCurve->input("frequencyBands").set(_pool.value<vector<vector<Real> > >("internal.frequencyBands"));
  _noveltyCurve->output("novelty").set(novelty);
  _noveltyCurve->compute();

  for (size_t i=0; i<novelty.size(); ++i) {
    _novelty.push(novelty[i]);
  }
  return FINISHED;
}


} // namespace streaming
} // namespace essentia