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

#include <iostream>
#include <fstream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h>
#include <essentia/pool.h>
#include "credit_libav.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;

int main(int argc, char* argv[]) {

  // register the algorithms in the factory(ies)
  essentia::init();

  /////// PARAMS //////////////

  // we want to compute the MFCC of a file: we need the create the following:
  // audioloader -> framecutter -> windowing -> FFT -> MFCC

  AlgorithmFactory& factory = standard::AlgorithmFactory::instance();

  Algorithm* csm = factory.create("CrossSimilarityMatrix");

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos ---------" << endl;

  vector<vector<Real> > queryFeatureinput(1000, vector<Real>(12, 1));
  vector<vector<Real> > referenceFeatureinput(2000, vector<Real>(12, 1));
  vector<vector<Real> > csmout;
  csm->input("queryFeature").set(queryFeatureinput);
  csm->input("referenceFeature").set(referenceFeatureinput);
  csm->output("csm").set(csmout);

  /////////// STARTING THE ALGORITHMS //////////////////

  csm->compute();
  cout << "Output matrix size: " << csmout.size() << "\t" << csmout[0].size() << endl;
  delete csm;

  essentia::shutdown();

  return 0;
}
