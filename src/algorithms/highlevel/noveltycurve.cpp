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

#include "algorithmfactory.h"
#include "essentiamath.h"
#include "noveltycurve.h"

using namespace std;

namespace essentia {
namespace standard {

const char* NoveltyCurve::name = "NoveltyCurve";
const char* NoveltyCurve::description = DOC(
"Given an audio signal, this algorithm computes the novelty curve, such as defined in [1].\n"
"\n"
"References:\n"
"  [1] P. Grosche and M. Müller, \"A mid-level representation for capturing\n"
"  dominant tempo and pulse information in music recordings,\" in\n"
"  International Society for Music Information Retrieval Conference\n"
"  (ISMIR’09), 2009, pp. 189–194.");


vector<Real> NoveltyCurve::weightCurve(int size) {
  vector<Real> result(size, 0.0);
  int halfSize = size/2;
  int sqrHalfSize = halfSize*halfSize;
  int sqrSize = size*size;

  // NB:some of these curves have a +1 so we don't reach zero!!
  switch(_type) {
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
 * Returns a vector of as many values as were on the input. As we are derivating, the first
 * coefficient can't be computed and thus will be set to 0.
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

  // substract local mean
  for (int i=0; i<dsize; i++) {
    int start = i - meanSize/2, end = i + meanSize/2;
    // TODO: decide on which option to choose
    if (false) {
      // nico adjust
      start = max(start, 0);
      end = min(end, size-1);
    }
    else {
      // edu adjust
      //int dsize = size-1;
      if (start<0 && end>=dsize) {start=0; end=dsize;}
      else {
        if (start<0) { start=0; end=meanSize;}
        if (end>=dsize) { end=dsize; start=dsize-meanSize;}
      }
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


  //return novelty;
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
  /////////////////////////////////////////////////////////////////////////////
  // TODO: By trial-&-error I found that combining weightings (flat, quadratic,
  // linear and inverse quadratic) was giving better results. Should this be
  // left as is or should we allow the algorithm to work with the given
  // weightings from the configuration. This overrides the parameters, so if
  // left as is, they should be removed as well.
  /////////////////////////////////////////////////////////////////////////////
  _type = FLAT;
  vector<Real> aweights = weightCurve(nBands);
  _type = QUADRATIC;
  vector<Real> bweights = weightCurve(nBands);
  _type = LINEAR;
  vector<Real> cweights = weightCurve(nBands);
  _type = INVERSE_QUADRATIC;
  vector<Real> dweights = weightCurve(nBands);
  //sum novelty on all bands (weighted) to get a single novelty value per frame
  noveltyBands = essentia::transpose(noveltyBands); // back to [frames x bands]
  vector<Real> bnovelty(nFrames-1, 0.0);
  vector<Real> cnovelty(nFrames-1, 0.0);
  vector<Real> dnovelty(nFrames-1, 0.0);
  for (int frameIdx=0; frameIdx<nFrames-1; frameIdx++) { // nFrames -1 as noveltyBands is a derivative whose size is nframes-1
    const vector<Real>& frame = noveltyBands[frameIdx];
    for (int bandIdx=0; bandIdx<nBands; bandIdx++) {
      novelty[frameIdx] += aweights[bandIdx] * frame[bandIdx];
      bnovelty[frameIdx] += bweights[bandIdx] * frame[bandIdx];
      cnovelty[frameIdx] += cweights[bandIdx] * frame[bandIdx];
      dnovelty[frameIdx] += dweights[bandIdx] * frame[bandIdx];
    }
  }
  for (int frameIdx=0; frameIdx<nFrames-1; frameIdx++) {
      novelty[frameIdx] *= bnovelty[frameIdx];
      novelty[frameIdx] *= cnovelty[frameIdx];
      novelty[frameIdx] *= dnovelty[frameIdx];
  }


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
