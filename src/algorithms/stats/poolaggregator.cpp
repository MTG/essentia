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

#include "poolaggregator.h"
#include "algorithmfactory.h"
#include "essentiamath.h"
#include "essentiautil.h"
#include "tnt/tnt2essentiautils.h"

using namespace std;
using namespace essentia;
using namespace standard;

const char* PoolAggregator::name = "PoolAggregator";
const char* PoolAggregator::description = DOC("This algorithm performs statistical aggregation on a Pool and places the results of the aggregation into a new Pool. Supported statistical units are:\n"
  "\t'min' (minimum),\n"
  "\t'max' (maximum),\n"
  "\t'median'\n"
  "\t'mean'\n"
  "\t'var' (variance),\n"
  "\t'skew' (skewness),\n"
  "\t'kurt' (kurtosis),\n"
  "\t'dmean' (mean of the derivative),\n"
  "\t'dvar' (variance of the derivative),\n"
  "\t'dmean2' (mean of the second derivative),\n"
  "\t'dvar2' (variance of the second derivative),\n"
  "\t'cov' (covariance), and\n"
  "\t'icov' (inverse covariance).\n"
  "\t'copy' (verbatim copy of descriptor, no aggregation; exclusive: cannot be performed with any other statistical units).\n"
  "\t'value' (copy of the descriptor, but the value is placed under the name '<descriptor name>.value')\n\n"

  "These statistics can be computed for single dimensional vectors in a Pool, with the exception of 'cov' and 'icov'. All of the above statistics can be\n"
  "computed for two dimensional vectors in the Pool. With the exception of 'cov' and 'icov', two-dimensional statistics are calculated by aggregating\n"
  "each column and placing the result into a vector of the same size as the size of each vector in the input Pool. The previous implies that each\n"
  "vector in the pool (under a particular descriptor of course) must have equal size. This implication also applies for 'cov' and 'icov'.\n\n"

  "An additional restriction for using the 'icov' statistic is that the covariance matrix for a particular descriptor must be invertible. The 'cov' and 'icov' aggregation statistics each return a square matrix with dimension equal to the length of the vectors under the given descriptor.\n\n"

  "Please also note that only the absolute values of the first and second derivates are considered when calculating the mean ('dmean' and 'dmean2') as well as for the variance ('dvar' and 'dvar2'). This is to avoid a trivial solution for the mean.");


// initialize supported statistics set
const char* supportedStats[] =
  {"min", "max", "median", "mean", "var", "skew", "kurt",
   "dmean", "dvar", "dmean2", "dvar2",
   "cov", "icov",
   "copy", "value"};
vector<string> tmp = arrayToVector<string>(supportedStats);
const set<string> PoolAggregator::_supportedStats(tmp.begin(), tmp.end());

void addMatrixAsVectorVector(Pool& p, const string& key, const TNT::Array2D<Real>& mat) {
  for (int i=0; i<int(mat.dim1()); ++i) {
    vector<Real> v(mat.dim1());
    for (int j=0; j<int(mat.dim2()); ++j) {
      v[j] = mat[i][j];
    }
    p.add(key, v);
  }
}


void PoolAggregator::aggregateSingleRealPool(const Pool& input, Pool& output) {
  const map<string, Real>& realPool = input.getSingleRealPool();

  for (map<string,Real>::const_iterator it = realPool.begin();
       it != realPool.end();
       ++it) {
    string key = it->first;
    Real data = it->second;
    output.set(key, data);
  }
}

void PoolAggregator::aggregateRealPool(const Pool& input, Pool& output) {
  PoolOf(Real) realPool = input.getRealPool();

  for (PoolOf(Real)::const_iterator it = realPool.begin();
       it != realPool.end();
       ++it) {
    string key = it->first;
    vector<Real> data = it->second;
    int dsize = int(data.size());

    // mean and variance
    Real meanVal = mean(data);
    Real varianceVal = variance(data, meanVal);

    // median
    Real medianVal = median(data);

    // skewness and kurtosis
    Real skewnessVal = skewness(data, meanVal);
    Real kurtosisVal = kurtosis(data, meanVal);

    // min and max
    Real minVal = data[0], maxVal = data[0];
    for (int i=1; i<dsize; ++i) {
      minVal = min(minVal, data[i]);
      maxVal = max(maxVal, data[i]);
    }

    // derived mean & var
    vector<Real> derived(dsize > 1 ? dsize-1 : 1, 0.0);
    vector<Real> derived2(dsize > 2 ? dsize-2 : 1, 0.0);

    for (int i=0; i<dsize-1; ++i) {
      derived[i] = data[i+1] - data[i];
    }
    for (int i=0; i<dsize-2; ++i) {
      derived2[i] = derived[i+1] - derived[i];
    }

    Real dmeanVal, d2meanVal, dvarianceVal, d2varianceVal;

    // we need to perform the absolute value conversion before taking the
    // variance so that the mean and variance caclulation both use the absolute
    // value technique and thus consistent
    for (int i=0; i<(int)derived.size(); i++) derived[i] = abs(derived[i]);
    for (int i=0; i<(int)derived2.size(); i++) derived2[i] = abs(derived2[i]);
    dmeanVal = mean(derived);
    d2meanVal = mean(derived2);
    dvarianceVal = variance(derived, mean(derived));
    d2varianceVal = variance(derived2, mean(derived2));

    // figure out which computed stats to add to the output pool
    const vector<string>& stats = getStats(key);
    for (int i=0; i<(int)stats.size(); ++i) {
      if      (stats[i] == "mean")   output.set(key + ".mean", meanVal);
      else if (stats[i] == "median") output.set(key + ".median", medianVal);
      else if (stats[i] == "min")    output.set(key + ".min", minVal);
      else if (stats[i] == "max")    output.set(key + ".max", maxVal);
      else if (stats[i] == "var")    output.set(key + ".var", varianceVal);
      else if (stats[i] == "skew")   output.set(key + ".skew", skewnessVal);
      else if (stats[i] == "kurt")   output.set(key + ".kurt", kurtosisVal);
      else if (stats[i] == "dmean")  output.set(key + ".dmean", dmeanVal);
      else if (stats[i] == "dvar")   output.set(key + ".dvar", dvarianceVal);
      else if (stats[i] == "dmean2") output.set(key + ".dmean2", d2meanVal);
      else if (stats[i] == "dvar2")  output.set(key + ".dvar2", d2varianceVal);
      else if (stats[i] == "copy") {
        for (int i=0; i<int(data.size()); ++i) {
          output.add(key, data[i]);
        }
      }
      else if (stats[i] == "value") {
        string subkey = key + ".value";
        for (int i=0; i<int(data.size()); ++i) {
          output.add(subkey, data[i]);
        }
      }
    }
  }
}

void PoolAggregator::aggregateSingleVectorRealPool(const Pool& input, Pool& output) {
  const map<string, vector<Real> >& vectorRealPool = input.getSingleVectorRealPool();
  for (map<string, vector<Real> >::const_iterator it = vectorRealPool.begin();
       it != vectorRealPool.end();
       ++it) {

    string key = it->first;
    vector<Real> data = it->second;
    output.set(key, data);
  }
}

void PoolAggregator::aggregateVectorRealPool(const Pool& input, Pool& output) {
  PoolOf(vector<Real>) vectorRealPool = input.getVectorRealPool();

  for (PoolOf(vector<Real>)::const_iterator it = vectorRealPool.begin();
       it != vectorRealPool.end();
       ++it) {

    string key = it->first;
    vector<vector<Real> > data = it->second;
    int dsize = data.size();

    if (dsize == 0) continue;

    // if pool value consists of only one vector, don't perform aggregation,
    // just add it to the output
    if (dsize == 1) {
      output.add(key, data[0]);
      continue;
    }

    int vsize = data[0].size();

    // check if all the vectors are the same size, otherwise skip the descriptor
    bool skipDescriptor = false;
    for (int i=1; i<dsize; ++i) {
      if ((int)data[i].size() != vsize) {
        cout << "WARNING: PoolAggregator: not aggregating \"" << key << "\" because it has frames of different sizes" << endl;
        skipDescriptor = true;
        break;
      }
    }
    if (skipDescriptor) continue;


    // mean & var
    vector<Real> meanVals = meanFrames(data);
    vector<Real> varVals = varianceFrames(data);

    // median
    vector<Real> medianVals = medianFrames(data);

    // skewness & kurtosis
    vector<Real> skewnessVals = skewnessFrames(data);
    vector<Real> kurtosisVals = kurtosisFrames(data);

    // min & max
    vector<Real> minVals(vsize, 0.0), maxVals(vsize, 0.0);
    for (int j=0; j<vsize; j++) minVals[j] = maxVals[j] = data[0][j]; // init values
    for (int i=1; i<dsize; i++) {
      for (int j=0; j<vsize; j++) {
        minVals[j] = min(data[i][j], minVals[j]);
        maxVals[j] = max(data[i][j], maxVals[j]);
      }
    }

    // derived mean & var
    vector<vector<Real> > derived(dsize > 1 ? dsize-1 : 1, vector<Real>(vsize, 0.0));
    vector<vector<Real> > derived2(dsize > 2 ? dsize-2 : 1, vector<Real>(vsize, 0.0));

    // first derivative
    for (int i=0; i<dsize-1; i++) {
      for (int j=0; j<vsize; j++) {
        derived[i][j] += data[i+1][j] - data[i][j];
      }
    }

    // second derivative
    for (int i=0; i<dsize-2; i++) {
      for (int j=0; j<vsize; j++) {
        derived2[i][j] += derived[i+1][j] - derived[i][j];
      }
    }

    for (int i=0; i<int(derived.size()); i++) {
      for (int j=0; j<int(derived[i].size()); j++) {
        derived[i][j] = abs(derived[i][j]);
      }
    }

    for (int i=0; i<int(derived2.size()); i++) {
      for (int j=0; j<int(derived2[i].size()); j++) {
        derived2[i][j] = abs(derived2[i][j]);
      }
    }

    vector<Real> dmeanVals = meanFrames(derived);
    vector<Real> d2meanVals = meanFrames(derived2);
    vector<Real> dvarVals = varianceFrames(derived);
    vector<Real> d2varVals = varianceFrames(derived2);

    // only compute cov and icov matrix if asked, because it could throw an
    // exception if matrix is singular...
    const vector<string>& stats = getStats(key);

    vector<vector<Real> > cov(vsize), icov(vsize);

    if (contains(stats, string("cov")) || contains(stats, string("icov"))) {

      // create an Array2D and copy all the data values into it
      TNT::Array2D<Real> frames(dsize, vsize);
      for (int i=0; i<dsize; i++) {
        for (int j=0; j<vsize; j++) {
          frames[i][j] = data[i][j];
        }
      }

      vector<Real> framesMean; // not used
      TNT::Array2D<Real> covTnt, icovTnt;

      Algorithm* sg = AlgorithmFactory::create("SingleGaussian");
      sg->input("matrix").set(frames);
      sg->output("mean").set(framesMean);
      sg->output("covariance").set(covTnt);
      sg->output("inverseCovariance").set(icovTnt);

      sg->compute();

      delete sg;

      // convert the Array2D back into vector<vector<Real> >
      //for (int i=0; i<dsize; ++i) {
      int covSize = covTnt.dim1();
      for (int i=0; i<covSize; ++i) {
        cov[i].resize(covSize);
        icov[i].resize(covSize);
        for (int j=0; j<covSize; ++j) {
          cov[i][j] = covTnt[i][j];
          icov[i][j] = icovTnt[i][j];
        }
      }
    }

    // Now add all the computed statistics into the output pool
    for (int i=0; i<(int)stats.size(); ++i) {
      string subkey = key + "." + stats[i];

      if (stats[i] == "mean")
        for (int j=0; j<int(meanVals.size()); ++j) output.add(subkey, meanVals[j]);

      else if (stats[i] == "median")
        for (int j=0; j<int(medianVals.size()); ++j) output.add(subkey, medianVals[j]);
    
      else if (stats[i] == "min")
        for (int j=0; j<int(minVals.size()); ++j) output.add(subkey, minVals[j]);

      else if (stats[i] == "max")
        for (int j=0; j<int(maxVals.size()); ++j) output.add(subkey, maxVals[j]);

      else if (stats[i] == "var")
        for (int j=0; j<int(varVals.size()); ++j) output.add(subkey, varVals[j]);

      else if (stats[i] == "skew")
        for (int j=0; j<int(skewnessVals.size()); ++j) output.add(subkey, skewnessVals[j]);

      else if (stats[i] == "kurt")
        for (int j=0; j<int(kurtosisVals.size()); ++j) output.add(subkey, kurtosisVals[j]);

      else if (stats[i] == "dmean")
        for (int j=0; j<int(dmeanVals.size()); ++j) output.add(subkey, dmeanVals[j]);

      else if (stats[i] == "dvar")
        for (int j=0; j<int(dvarVals.size()); ++j) output.add(subkey, dvarVals[j]);

      else if (stats[i] == "dmean2")
        for (int j=0; j<int(d2meanVals.size()); ++j) output.add(subkey, d2meanVals[j]);

      else if (stats[i] == "dvar2")
        for (int j=0; j<int(d2varVals.size()); ++j) output.add(subkey, d2varVals[j]);

      else if (stats[i] == "cov")
        for (int j=0; j<vsize; ++j) output.add(subkey, cov[j]);

      else if (stats[i] == "icov")
        for (int j=0; j<vsize; ++j) output.add(subkey, icov[j]);

      else if (stats[i] == "copy")
        // don't use the subkey in this case, just key
        for (int j=0; j<int(data.size()); ++j) output.add(key, data[j]);

      else if (stats[i] == "value")
        for (int j=0; j<int(data.size()); ++j) output.add(subkey, data[j]);
    }
  }
}


void PoolAggregator::aggregateSingleStringPool(const Pool& input, Pool& output) {
  const map<string, string>&  stringPool = input.getSingleStringPool();
  for (map<string, string>::const_iterator it = stringPool.begin();
       it != stringPool.end();
       ++it) {
    string key = it->first;
    string data = it->second;
    output.set(key, data);
  }
}

void PoolAggregator::aggregateStringPool(const Pool& input, Pool& output) {
  const PoolOf(string)& stringPool = input.getStringPool();

  for (PoolOf(string)::const_iterator it = stringPool.begin();
       it != stringPool.end();
       ++it) {
    string key = it->first;
    vector<string> data = it->second;

    for (int i=0; i<(int)data.size(); ++i) {
      output.add(key, data[i]);
    }
  }
}


void PoolAggregator::aggregateVectorStringPool(const Pool& input, Pool& output) {
  const PoolOf(vector<string>)& vectorStringPool = input.getVectorStringPool();

  for (PoolOf(vector<string>)::const_iterator it = vectorStringPool.begin();
       it != vectorStringPool.end();
       ++it) {
    string key = it->first;
    vector<vector<string> > data = it->second;

    for (int i=0; i<(int)data.size(); ++i) {
      output.add(key, data[i]);
    }
  }
}

void PoolAggregator::aggregateArray2DRealPool(const Pool& input, Pool& output) {
  PoolOf(TNT::Array2D<Real>) Array2DRealPool = input.getArray2DRealPool();

  for (PoolOf(TNT::Array2D<Real>)::const_iterator it = Array2DRealPool.begin();
       it != Array2DRealPool.end();
       ++it) {

    string key = it->first;
    vector<TNT::Array2D<Real> > data = it->second;
    // get frames:
    int dsize = data.size();

    if (dsize == 0) continue;

    // if pool value consists of only one vector, don't perform aggregation,
    // just add it to the output
    if (dsize == 1) {
      output.add(key, data[0]);
      continue;
    }

    // get the size of the data for each frame:
    int dim1 = data[0].dim1();
    int dim2 = data[0].dim2();

    // check if all the matrices have the same size, otherwise skip the descriptor
    bool skipDescriptor = false;
    for (int i=1; i<dsize; ++i) {
      if (data[i].dim1() != dim1 || data[i].dim2() != dim2) {
        cout << "WARNING: PoolAggregator: not aggregating \"" << key << "\" because it has frames of different sizes" << endl;
        skipDescriptor = true;
        break;
      }
    }
    if (skipDescriptor) continue;

    // mean:
    TNT::Array2D<Real> meanMat = meanMatrix(data);
    // var:
    TNT::Array2D<Real> varMat = varianceMatrix(data, meanMat);

    // min & max: computes the minimum/maximum number at each position:
    TNT::Array2D<Real> minMat(dim1, dim2), maxMat(dim1, dim2);
    for (int row=0; row<dim1; row++) {
      for (int col=0; col<dim2; col++) {
        minMat[row][col] = maxMat[row][col] = data[0][row][col]; // init values
      }
    }
    for (int i=1; i<dsize; i++) {
      for (int row=0; row<dim1; row++) {
        for (int col=0; col<dim2; col++) {
          minMat[row][col] = min(data[i][row][col], minMat[row][col]);
          maxMat[row][col] = max(data[i][row][col], maxMat[row][col]);
        }
      }
    }

    // derived mean & var
    vector<TNT::Array2D<Real>* > derived(dsize > 1 ? dsize-1 : 1);
    vector<TNT::Array2D<Real>* > derived2(dsize > 2 ? dsize-2 : 1);

    // first derivative
    for (int i=0; i<dsize-1; i++) {
      derived[i] = new TNT::Array2D<Real>(data[i+1] - data[i]);
    }

    // second derivative
    for (int i=0; i<dsize-2; i++) {
        derived2[i] = new TNT::Array2D<Real>(*derived[i+1] - *derived[i]);
    }

    for (int i=0; i<int(derived.size()); i++) {
      for (int row=0; row<derived[i]->dim1(); row++) {
        for (int col=0; col<derived[i]->dim2(); col++) {
          (*derived[i])[row][col] = abs((*derived[i])[row][col]);
        }
      }
    }

    // this could be done in the nested for-loops from above.. should we?
    for (int i=0; i<int(derived2.size()); i++) {
      for (int row=0; row<derived2[i]->dim1(); row++) {
        for (int col=0; col<derived2[i]->dim2(); col++) {
          (*derived2[i])[row][col] = abs((*derived2[i])[row][col]);
        }
      }
    }

    TNT::Array2D<Real> dmeanMat = meanMatrix(derived);
    TNT::Array2D<Real> d2meanMat = meanMatrix(derived2);
    TNT::Array2D<Real> dvarMat = varianceMatrix(derived, meanMatrix(derived));
    TNT::Array2D<Real> d2varMat = varianceMatrix(derived2, meanMatrix(derived2));

    for (int i=0; i<(int)derived.size(); i++) delete derived[i];
    for (int i=0; i<(int)derived2.size(); i++) delete derived2[i];

    // cov and icov matrix : not implemented yet
    const vector<string>& stats = getStats(key);

    if (contains(stats, string("cov")) || contains(stats, string("icov"))) {
      cout << "PoolAggregator: Covariance and inverse covariance for vectors of matrices are not yet implemented" << endl;
    }

    // Now add all the computed statistics into the output pool
    for (int i=0; i<int(stats.size()); ++i) {
      string subkey = key + "." + stats[i];
      if (stats[i] == "mean") addMatrixAsVectorVector(output, subkey, meanMat);
      else if (stats[i] == "median") { /* TODO: not implemented */ }
      else if (stats[i] == "min") addMatrixAsVectorVector(output, subkey, minMat);
      else if (stats[i] == "max") addMatrixAsVectorVector(output, subkey, maxMat);
      else if (stats[i] == "var") addMatrixAsVectorVector(output, subkey, varMat);
      else if (stats[i] == "dmean") addMatrixAsVectorVector(output, subkey, dmeanMat);
      else if (stats[i] == "dvar") addMatrixAsVectorVector(output, subkey, dvarMat);
      else if (stats[i] == "dmean2") addMatrixAsVectorVector(output, subkey, d2meanMat);
      else if (stats[i] == "dvar2") addMatrixAsVectorVector(output, subkey, d2varMat);
      else if (stats[i] == "cov") { /* TODO: not implemented */ }
      else if (stats[i] == "icov") { /* TODO: not implemented */ }
      else if (stats[i] == "copy")
        for (int j=0; j<int(data.size()); ++j) output.add(key, data[j]);
      else if (stats[i] == "value")
        for (int j=0; j<int(data.size()); ++j) output.add(subkey, data[j]);
    }
  }
}

void PoolAggregator::configure() {
  _defaultStats = parameter("defaultStats").toVectorString();
  _exceptions = parameter("exceptions").toMapVectorString();

  // if the default stats includes the 'copy' statistical unit, make sure it
  // is the only one
  if (indexOf<string>(_defaultStats, "copy") != -1 &&
      int(_defaultStats.size()) != 1) {
    throw EssentiaException("PoolAggregator: the 'copy' aggregation statistic "
                            "is exclusive, it cannot be used with other "
                            "statistics for the same descriptor");
  }

  // make sure there are no unsupported statistics in 'defaultStats'
  for (int i=0; i<(int)_defaultStats.size(); ++i) {
    if (_supportedStats.find(_defaultStats[i]) == _supportedStats.end()) {
      throw EssentiaException("PoolAggregator: unsupported aggregation statistic: '" + _defaultStats[i] + "'");
    }
  }

  // make sure there are no duplicate keys in 'exceptions'
  // make sure there are no unsupported statistics in the values of 'exceptions'
  set<string> keys;
  for (map<string, vector<string> >::const_iterator it = _exceptions.begin();
       it != _exceptions.end();
       ++it) {
    const vector<string>& exceptionStats = it->second;

    if (indexOf<string>(exceptionStats, "copy") != -1 &&
        int(exceptionStats.size()) != 1) {
      throw EssentiaException("PoolAggregator: the 'copy' aggregation statistic "
                              "is exclusive, it cannot be used with other "
                              "statistics for the same descriptor");
    }

    for (int i=0; i<(int)exceptionStats.size(); ++i) {
      if (_supportedStats.find(exceptionStats[i]) == _supportedStats.end()) {
        throw EssentiaException("PoolAggregator: unsupported aggregation statistic: '" + exceptionStats[i] + "'");
      }
    }
  }
}


void PoolAggregator::compute() {
  const Pool& input = _input.get();
  Pool& output = _output.get();
  aggregateSingleRealPool(input, output);
  aggregateRealPool(input, output);
  aggregateSingleVectorRealPool(input, output);
  aggregateVectorRealPool(input, output);
  aggregateStringPool(input, output);
  aggregateSingleStringPool(input, output);
  aggregateVectorStringPool(input, output);
  aggregateArray2DRealPool(input, output);
}


const vector<string>& PoolAggregator::getStats(const string& key) const {
  if (_exceptions.count(key) > 0) {
    return (*(_exceptions.find(key))).second;
  }
  else {
    return _defaultStats;
  }
}
