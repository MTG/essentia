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

#include "streaming_extractorpostprocess.h"
#include "poolstorage.h"
#include "algorithmfactory.h"
#include "essentiamath.h" // for meanFrames and meanVariances

using namespace std;
using namespace essentia;
using namespace standard;

void PCA(Pool& pool, const string& nspace) {

  // namespace:
  string llspace = "lowlevel.";
  if (!nspace.empty()) llspace = nspace + ".lowlevel.";

  vector<vector<Real> > sccoeffs = pool.value<vector<vector<Real> > >(llspace + "sccoeffs");
  vector<vector<Real> > scvalleys = pool.value<vector<vector<Real> > >(llspace + "scvalleys");

  Pool poolSc, poolTransformed;

  for (int iFrame = 0; iFrame < (int)sccoeffs.size(); iFrame++) {
    vector<Real> merged(2*sccoeffs[iFrame].size(), 0.0);
    for(int i=0, j=0; i<(int)sccoeffs[iFrame].size(); i++, j++) {
      merged[j++] = sccoeffs[iFrame][i];
      merged[j] = scvalleys[iFrame][i];
    }
    poolSc.add("contrast", merged);
  }

  Algorithm* pca  = AlgorithmFactory::create("PCA",
                                             "namespaceIn",  "contrast",
                                             "namespaceOut", "contrast");
  pca->input("poolIn").set(poolSc);
  pca->output("poolOut").set(poolTransformed);
  pca->compute();

  pool.set(llspace + "spectral_contrast.mean",
           meanFrames(poolTransformed.value<vector<vector<Real> > >("contrast")));
  pool.set(llspace + "spectral_contrast.var",
           varianceFrames(poolTransformed.value<vector<vector<Real> > >("contrast")));

  // remove original data from spectral contrast:
  pool.remove(llspace + "sccoeffs");
  pool.remove(llspace + "scvalleys");
  delete pca;
}

// Add missing descriptors which are not computed yet, but will be for the
// final release or during the 1.x cycle. However, the schema need to be
// complete before that, so just put default values for these.
// Also make sure that some descriptors that might have fucked up come out nice.
void PostProcess(Pool& pool, const Pool& options, const string& nspace) {
  if (options.value<Real>("rhythm.compute") != 0) {
    string rhythmspace = "rhythm.";
    if (!nspace.empty()) rhythmspace = nspace + ".rhythm.";
    const vector<string>& descNames = pool.descriptorNames();
    if (find(descNames.begin(), descNames.end(), rhythmspace + "bpm_confidence") == descNames.end())
      pool.set(rhythmspace + "bpm_confidence", 0.0);
    if (find(descNames.begin(), descNames.end(), rhythmspace + "perceptual_tempo") == descNames.end())
      pool.set(rhythmspace + "perceptual_tempo", "unknown");
    if (find(descNames.begin(), descNames.end(), rhythmspace + "beats_loudness") == descNames.end())
      pool.add(rhythmspace + "beats_loudness", Real(0.0));
    if (find(descNames.begin(), descNames.end(), rhythmspace + "beats_loudness_band_ratio") == descNames.end())
      pool.add(rhythmspace + "beats_loudness_band_Ratio", vector<Real>());
    if (find(descNames.begin(), descNames.end(), rhythmspace + "rubato_start") == descNames.end())
      pool.set(rhythmspace + "rubato_start", vector<Real>(0));
    if (find(descNames.begin(), descNames.end(), rhythmspace + "rubato_stop") == descNames.end())
      pool.set(rhythmspace + "rubato_stop", vector<Real>(0));
  }
  // PCA analysis of spectral contrast output:
  PCA(pool, nspace);
}

