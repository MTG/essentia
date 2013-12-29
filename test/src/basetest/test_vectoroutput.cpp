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
  Vector<int> output;
  int array[] = {1, 2, 3, 4};
  VectorInput<int>* gen = new VectorInput<int>(array);
  connect(gen->output("data"), output);
  scheduler::Network(gen).run();

  Vector<int> expected = arrayToVector<int>(array);
  EXPECT_VEC_EQ(output, expected);
}

TEST(VectorOutput, Real) {
  Vector<Real> output;
  Real array[] = {1.1, 2.2, 3.3, 4.4};
  VectorInput<Real>* gen = new VectorInput<Real>(array);
  connect(gen->output("data"), output);
  scheduler::Network(gen).run();

  Vector<Real> expected = arrayToVector<Real>(array);
  EXPECT_VEC_EQ(output, expected);
  }

TEST(VectorOutput, String) {
  Vector<string> output;
  const char* array[] = {"foo", "bar", "foo-bar"};
  VectorInput<string>* gen = new VectorInput<string>(array);
  connect(gen->output("data"), output);
  scheduler::Network(gen).run();

  Vector<string> expected = arrayToVector<string>(array);
  EXPECT_VEC_EQ(output, expected);
}

TEST(VectorOutput, VectorInt) {
  Vector<Vector<int> > output;
  int size = 3;
  Vector<Vector<int> > v(size, Vector<int>(size));
  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      v[i][j] = rand();
    }
  }

  VectorInput<Vector<int> >* gen = new VectorInput<Vector<int> >(&v);
  connect(gen->output("data"), output);
  scheduler::Network(gen).run();
  EXPECT_MATRIX_EQ(output, v);
}

TEST(VectorOutput, VectorReal) {
  Vector<Vector<Real> > output;
  int size = 3;
  Vector<Vector<Real> > v(size, Vector<Real>(size));
  for (int i=0; i<size; i++) {
    for (int j=0; j<size; j++) {
      v[i][j] = Real(rand()/Real(RAND_MAX));
    }
  }

  VectorInput<Vector<Real> >* gen = new VectorInput<Vector<Real> >(&v);
  connect(gen->output("data"), output);
  scheduler::Network(gen).run();
  EXPECT_MATRIX_EQ(output, v);
}
