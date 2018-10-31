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
#include <boost/multi_array.hpp>
#include "credit_libav.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;

int main(int argc, char* argv[]) {

  // register the algorithms in the factory(ies)
  essentia::init();

  Pool pool;

  // we want to compute the MFCC of a file: we need the create the following:
  // audioloader -> framecutter -> windowing -> FFT -> MFCC

  AlgorithmFactory& factory = standard::AlgorithmFactory::instance();

    const char* inputNames[] = { "bidirectional_1_input_1" };
    const char* outputNames[] = { "output_node0", "activation_1_1/Sigmoid" };

    std::vector<std::string> inputNamesVector = arrayToVector<std::string>(inputNames);
    std::vector<std::string> outputNamesVector = arrayToVector<std::string>(outputNames);

  Algorithm* TFP = factory.create("TensorflowPredict",
                                  "inputs", inputNamesVector,  
                                  "outputs", outputNamesVector);

  /////////// CONNECTING THE ALGORITHMS ////////////////
  cout << "-------- connecting algos ---------" << endl;

  // Audio -> FrameCutter
  boost::multi_array<Real, 3> input(boost::extents[5][4000][90]);
  pool.add("bidirectional_1_input_1", input);
  // boost::multi_array<Real, 3> output;


  TFP->input("poolIn").set(pool);
  TFP->output("poolOut").set(pool);


  /////////// STARTING THE ALGORITHMS //////////////////
  cout << "-------- start processing --------" << endl;

  TFP->compute();

  // E_INFO(output[0][0][0]);
  // E_INFO(output[2][0][0]);
  // E_INFO(output[0][2][0]);
  // E_INFO(output[0][0][2]);

  delete TFP;

  essentia::shutdown();
  return 0;
}
