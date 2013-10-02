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

#include "gaiatransform.h"
#include "essentia.h"
#include <gaia2/point.h>
#include <gaia2/convert.h>

using namespace std;
using namespace essentia;
using namespace standard;

const char* GaiaTransform::name = "GaiaTransform";
const char* GaiaTransform::description = DOC(
"Applies a given Gaia2 transformation history to a given pool. It first converts the pool into a gaia2::Point suitable for the history, applies the history, and converts back the resulting point into an essentia Pool. In particular, it allows classification.\n"
"\n"
"Note that in order to enable this algorithm it is necessary to install Gaia2 library before building Essentia.\n"
"\n"
"References:\n"
"  [1] Gaia - A library for similarity in high-dimensional spaces,\n"
"  http://github.com/MTG/gaia");



void GaiaTransform::configure() {
  string filename;
  try {
    filename = parameter("history").toString();
  }
  catch (...) {}

  if (filename.empty()) {
    _configured = false;
    return;
  }

  _history.load(QString::fromStdString(filename));

  _configured = true;
}



/**
 * Create a new point with the given layout and copy those values it needs from the
 * given pool.
 */
gaia2::Point* poolToPoint(const Pool& pool, const gaia2::PointLayout& layout) {
  gaia2::Point* result = new gaia2::Point(layout);

  foreach (const QString& d, layout.descriptorNames(gaia2::RealType)) {

    string dname = d.mid(1).toStdString();
    if (pool.contains<Real>(dname)) {
      // descriptor is a single value
      Real value = pool.value<Real>(dname);
      result->setValue(d, gaia2::RealDescriptor(value));
    }
    else if (pool.contains<vector<Real> >(dname)) {
      // descriptor is a vector of reals
      const vector<Real>& value = pool.value<vector<Real> >(dname);
      result->setValue(d, gaia2::RealDescriptor(value));
    }
    else if (pool.contains<vector<vector<Real> > >(dname)) {
      // descriptor is a matrix
      const vector<vector<Real> >& value = pool.value<vector<vector<Real> > >(d.mid(1).toStdString());

      // copy it in a single vector (flatten), such as what gaia uses
      int rows = value.size();
      int cols = value[0].size();

      gaia2::RealDescriptor vecmat(2 + rows*cols, 0.0);
      vecmat[0] = rows;
      vecmat[1] = cols;
      for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
          vecmat[2+i*cols+j] = value[i][j];
        }
      }
      result->setValue(d, vecmat);
    }
    else {
      throw EssentiaException("Descriptor ", dname, " could not be found in pool");
    }

  }

  // for string descs, we need to be looking at both the string and enum types in the layout
  QStringList stringDescs = layout.descriptorNames(gaia2::StringType) +
                            layout.descriptorNames(gaia2::EnumType);

  foreach (const QString& d, stringDescs) {
    // NB: we should only have single strings here
    // ignore those descriptors which are added by the YamlOutput before giving them to Gaia
    if (d.startsWith(".metadata.version")) continue;

    string dname = d.mid(1).toStdString();
    if (pool.contains<string>(dname)) {
      // descriptor is a single string
      string label = pool.value<string>(dname);
      result->setLabel(d, gaia2::StringDescriptor(QString::fromStdString(label)));
    }
    else if (pool.contains<vector<string> >(dname)) {
      // descriptor is a list of strings
      vector<string> label = pool.value<vector<string> >(dname);
      result->setLabel(d, gaia2::convert::VectorString_to_StringDescriptor(label));
    }
    else {
      throw EssentiaException("Descriptor ", dname, " could not be found in pool");
    }
  }

  return result;
}

void checkNotIn(const string& desc, const set<string>& alldescs) {
  if (alldescs.find(desc) != alldescs.end()) {
    throw EssentiaException("GaiaTransform is trying to overwrite the ", desc, " value, which is already present in the pool. You can't do that.");
  }
}

/**
 * Copy all the values from the point inside the pool.
 */
void pointToPool(const gaia2::Point* p, Pool& pool, const Pool& origPool) {
  const gaia2::PointLayout& layout = p->layout();
  vector<string> alldescsv = pool.descriptorNames();
  set<string> alldescs(alldescsv.begin(), alldescsv.end());

  foreach (const QString& d, layout.descriptorNames(gaia2::RealType)) {
    string pooldesc = d.mid(1).toStdString();
    checkNotIn(pooldesc, alldescs);

    gaia2::RealDescriptor desc = p->value(d);

    if (desc.size() == 1)    pool.set(pooldesc, desc[0]);
    else if (desc.isEmpty()) pool.set(pooldesc, vector<Real>());
    else                     pool.set(pooldesc, vector<Real>(&desc[0],
                                                             &desc[0]+desc.size()));

    // in fact, if desc.size == 1 and we're not in the single pool, we'd better
    // put back the value as a list of size 1
    const std::map<std::string, Real>& srpool = origPool.getSingleRealPool();

    if ((desc.size() == 1) && (srpool.find(pooldesc) == srpool.end())) {
      pool.remove(pooldesc);
      for (int i=0; i<desc.size(); i++) pool.add(pooldesc, desc[i]);
    }

    // if desc was a vector<vector<Real> >, it will have been converted to a single
    // vector<Real>, where the first 2 values are the dimensions. Construct this matrix back.
    const PoolOf(std::vector<Real>)& vrpool = origPool.getVectorRealPool();

    if (vrpool.find(pooldesc) != vrpool.end()) {
      int rows = int(desc[0]);
      int cols = int(desc[1]);

      // assert we're not doing something potentially stupid
      if ((rows*cols + 2) != desc.size()) {
        cout << "Oops, internal error in GaiaTransform::pointToPool, ignoring..." << endl;
        continue;
      }
      pool.remove(pooldesc);

      for (int i=0; i<rows; i++) {
        pool.add(pooldesc, vector<Real>(&desc[2+ i*cols],
                                        &desc[2+ i*cols] + cols));
      }
    }
  }

  // for string descs, we need to be looking at both the string and enum types in the layout
  QStringList stringDescs = layout.descriptorNames(gaia2::StringType) +
                            layout.descriptorNames(gaia2::EnumType);

  foreach (const QString& d, stringDescs) {
    string pooldesc = d.mid(1).toStdString();
    checkNotIn(pooldesc, alldescs);

    gaia2::StringDescriptor desc = p->label(d);

    // little check: see if we haven't removed the version before
    if (pooldesc == "metadata.version.essentia") {
      pool.set(pooldesc, essentia::version);
      continue;
    }

    if (desc.size() == 1) pool.set(pooldesc, desc[0].toStdString());
    else {
      for (int i=0; i<desc.size(); i++) {
        pool.add(pooldesc, desc[i].toStdString());
      }
    }
  }
}

void GaiaTransform::compute() {
  if (!_configured) {
    throw EssentiaException("GaiaTransform: Algorithm is not properly configured");
  }

  const Pool& inputPool = _inputPool.get();
  Pool& outputPool = _outputPool.get();

  gaia2::Point* p = poolToPoint(inputPool, _history.at(0).layout);
  gaia2::Point* result = _history.mapPoint(p, true); // p has been deleted by mapPoint

  // FIXME: should raise an exception if we overwrite a value which was previously in the pool
  pointToPool(result, outputPool, inputPool);

  // small hack: if we got an SVM transfo with associated probabilities, put them in
  // a nicer shape than the vector of anonymous reals it is.
  bool doesSVM = false;
  gaia2::ParameterMap params;
  for (int i=0; i<_history.size(); i++) {
    if (_history[i].analyzerName == "svmtrain") {
      params = _history[i].params;
      doesSVM = true;
      break;
    }
  }

  if (doesSVM && params.value("probability").toBool()) {
    // we need to remove the class and the probability vector and replace them with:
    // class:
    //   value: X
    //   probability: X
    //   all:
    //     cls1: X
    //     cls2: X
    //     ...
    QStringList classList = params.value("classMapping").toStringList();
    string className = params.value("className").toString().toStdString();
    string cls = outputPool.value<string>(className);
    vector<Real> probs = outputPool.value<vector<Real> >(className + "Probability");
    Q_ASSERT(classList.size() == (int)probs.size());

    outputPool.remove(className);
    outputPool.remove(className + "Probability");

    outputPool.set(className + ".value", cls);
    outputPool.set(className + ".probability", probs[classList.indexOf(QString::fromStdString(cls))]);
    for (int i=0; i<classList.size(); i++) {
      outputPool.set(className + ".all." + classList[i].toStdString(), probs[i]);
    }
  }

  delete result;
}


GaiaTransform::~GaiaTransform() {}
