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

#include "essentia_gtest.h"
#include "network.h"
#include "vectorinput.h"
#include "vectoroutput.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;


TEST(VectorOutput, Integer) {
  vector<int> output;
  int array[] = {1, 2, 3, 4};
  VectorInput<int>* gen = new VectorInput<int>(array);
  connect(gen->output("data"), output);
  scheduler::Network(gen).run();

  vector<int> expected = arrayToVector<int>(array);
  EXPECT_VEC_EQ(output, expected);
}

TEST(VectorOutput, Real) {
  vector<Real> output;
  Real array[] = {1.1, 2.2, 3.3, 4.4};
  VectorInput<Real>* gen = new VectorInput<Real>(array);
  connect(gen->output("data"), output);
  scheduler::Network(gen).run();

  vector<Real> expected = arrayToVector<Real>(array);
  EXPECT_VEC_EQ(output, expected);
  }

TEST(VectorOutput, String) {
  vector<string> output;
  const char* array[] = {"foo", "bar", "foo-bar"};
  VectorInput<string>* gen = new VectorInput<string>(array);
  connect(gen->output("data"), output);
  scheduler::Network(gen).run();

  vector<string> expected = arrayToVector<string>(array);
  EXPECT_VEC_EQ(output, expected);
}

TEST(VectorOutput, VectorInt) {
  vector<vector<int> > output;
  int size = 3;
  vector<vector<int> > v(size, vector<int>(size));
  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      v[i][j] = rand();
    }
  }

  VectorInput<vector<int> >* gen = new VectorInput<vector<int> >(&v);
  connect(gen->output("data"), output);
  scheduler::Network(gen).run();
  EXPECT_MATRIX_EQ(output, v);
}

TEST(VectorOutput, VectorReal) {
  vector<vector<Real> > output;
  int size = 3;
  vector<vector<Real> > v(size, vector<Real>(size));
  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      v[i][j] = Real(rand()/Real(RAND_MAX));
    }
  }

  VectorInput<vector<Real> >* gen = new VectorInput<vector<Real> >(&v);
  connect(gen->output("data"), output);
  scheduler::Network(gen).run();
  EXPECT_MATRIX_EQ(output, v);
}

