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

#include <algorithm>
#include "essentia_gtest.h"
using namespace std;
using essentia::Real;
using essentia::EssentiaException;


// Make sure the basic use case works
TEST(Pool, RealPoolSimple) {
  Real expectedVal = 6.9;
  vector<Real> expected;
  expected.push_back(expectedVal);

  essentia::Pool p;
  p.add("foo.bar", expectedVal);

  EXPECT_VEC_EQ(p.value<vector<Real> >("foo.bar"), expected);
}

// Make sure we can support having multiple values under one label
TEST(Pool, RealPoolMultiple) {
  Real expectedVal1 = 6.9;
  Real expectedVal2 = 16.0;
  vector<Real> expected;
  expected.push_back(expectedVal1);
  expected.push_back(expectedVal2);

  essentia::Pool p;
  p.add("foo.bar", expectedVal1);
  p.add("foo.bar", expectedVal2);

  EXPECT_VEC_EQ(p.value<vector<Real> >("foo.bar"), expected);
}

// Make sure we can support having multiple labels in the same pool
TEST(Pool, RealPoolMultipleLabels) {
  Real expectedVal1 = 6.9;
  Real expectedVal2 = 16.0;
  vector<Real> expected1;
  vector<Real> expected2;
  expected1.push_back(expectedVal1);
  expected2.push_back(expectedVal2);

  essentia::Pool p;
  p.add("foo.bar", expectedVal1);
  p.add("bar.foo", expectedVal2);

  EXPECT_VEC_EQ(p.value<vector<Real> >("foo.bar"), expected1);
  EXPECT_VEC_EQ(p.value<vector<Real> >("bar.foo"), expected2);
}

TEST(Pool, RealVectorPoolSimple) {
  vector<Real> expectedVec;
  expectedVec.push_back(1.6);
  expectedVec.push_back(0.9);
  expectedVec.push_back(19.85);

  vector<vector<Real> > expected;
  expected.push_back(expectedVec);

  essentia::Pool p;
  p.add("foo.bar", expectedVec);

  EXPECT_MATRIX_EQ(p.value<vector<vector<Real> > >("foo.bar"), expected);
}

TEST(Pool, RealVectorPoolMultiple) {
  vector<Real> expectedVec1;
  expectedVec1.push_back(1.6);
  expectedVec1.push_back(0.9);
  expectedVec1.push_back(19.85);

  vector<Real> expectedVec2;
  expectedVec2.push_back(-5.0);
  expectedVec2.push_back(0.0);
  expectedVec2.push_back(5.0);

  vector<vector<Real> > expected;
  expected.push_back(expectedVec1);
  expected.push_back(expectedVec2);

  essentia::Pool p;
  p.add("foo.bar", expectedVec1);
  p.add("foo.bar", expectedVec2);

  EXPECT_MATRIX_EQ(p.value<vector<vector<Real> > >("foo.bar"), expected);
}

TEST(Pool, RealVectorPoolMultipleLabels) {
  vector<Real> expectedVec1;
  expectedVec1.push_back(1.6);
  expectedVec1.push_back(0.9);
  expectedVec1.push_back(19.85);

  vector<Real> expectedVec2;
  expectedVec2.push_back(-5.0);
  expectedVec2.push_back(0.0);
  expectedVec2.push_back(5.0);

  vector<vector<Real> > expected1;
  expected1.push_back(expectedVec1);
  vector<vector<Real> > expected2;
  expected2.push_back(expectedVec2);

  essentia::Pool p;
  p.add("foo.bar", expectedVec1);
  p.add("bar.foo", expectedVec2);

  EXPECT_MATRIX_EQ(p.value<vector<vector<Real> > >("foo.bar"), expected1);
  EXPECT_MATRIX_EQ(p.value<vector<vector<Real> > >("bar.foo"), expected2);
}

// Test adding an empty vector
TEST(Pool, RealVectorEmpty) {
  vector<Real> emptyVec;
  vector<vector<Real> > expected;
  expected.push_back(emptyVec);

  essentia::Pool p;
  p.add("foo.bar", emptyVec);

  EXPECT_MATRIX_EQ(p.value<vector<vector<Real> > >("foo.bar"), expected);
}

// Make sure the lookup of a non-existant descriptorName fails
TEST(Pool, MissingDescriptorName) {
  essentia::Pool p;
  p.add("foo.bar", Real(0.0));

  ASSERT_THROW(p.value<vector<Real> >("bar.bar"), EssentiaException);
}

TEST(Pool, Remove) {
  Real expectedVal = 123.456;
  vector<Real> expected;
  expected.push_back(expectedVal);

  essentia::Pool p;
  p.add("foo.rab", expectedVal);
  p.add("foo.bar", Real(0.0));
  p.add("foo.bar", Real(1111.1111));
  p.remove("foo.bar");

  ASSERT_THROW(p.value<vector<Real> >("foo.bar"), EssentiaException);
  EXPECT_VEC_EQ(p.value<vector<Real> >("foo.rab"), expected);
}

// String type tests

TEST(Pool, StringPoolSimple) {
  vector<string> expected;
  expected.push_back("simple");

  essentia::Pool p;
  p.add("foo.bar", "simple");

  EXPECT_VEC_EQ(p.value<vector<string> >("foo.bar"), expected);
}

TEST(Pool, StringPoolMultiple) {
  string expectedVal1 = "mul";
  string expectedVal2 = "tiple";
  vector<string> expected;
  expected.push_back(expectedVal1);
  expected.push_back(expectedVal2);

  essentia::Pool p;
  p.add("foo.bar", expectedVal1);
  p.add("foo.bar", expectedVal2);

  EXPECT_VEC_EQ(p.value<vector<string> >("foo.bar"), expected);
}

TEST(Pool, StringPoolMultipleLabels) {
  string expectedVal1 = "multiple";
  string expectedVal2 = "labels";
  vector<string> expected1;
  vector<string> expected2;
  expected1.push_back(expectedVal1);
  expected2.push_back(expectedVal2);

  essentia::Pool p;
  p.add("foo.bar", expectedVal1);
  p.add("bar.foo", expectedVal2);

  EXPECT_VEC_EQ(p.value<vector<string> >("foo.bar"), expected1);
  EXPECT_VEC_EQ(p.value<vector<string> >("bar.foo"), expected2);
}

TEST(Pool, StringVectorPoolMultiple) {
  vector<string> expectedVec1;
  expectedVec1.push_back("1.6");
  expectedVec1.push_back("0.9");
  expectedVec1.push_back("19.85");

  vector<string> expectedVec2;
  expectedVec2.push_back("-5.0");
  expectedVec2.push_back("0.0");
  expectedVec2.push_back("5.0");

  vector<vector<string> > expected1;
  expected1.push_back(expectedVec1);
  vector<vector<string> > expected2;
  expected2.push_back(expectedVec2);

  essentia::Pool p;
  p.add("foo.bar", expectedVec1);
  p.add("bar.foo", expectedVec2);

  EXPECT_MATRIX_EQ(p.value<vector<vector<string> > >("foo.bar"), expected1);
  EXPECT_MATRIX_EQ(p.value<vector<vector<string> > >("bar.foo"), expected2);
}

// Test adding an empty vector
TEST(Pool, StringVectorEmpty) {
  vector<string> emptyVec;
  vector<vector<string> > expected;
  expected.push_back(emptyVec);

  essentia::Pool p;
  p.add("foo.bar", emptyVec);

  EXPECT_MATRIX_EQ(p.value<vector<vector<string> > >("foo.bar"), expected);
}

TEST(Pool, DescriptorNames) {
  string key1 = "foo.bar";
  string key2 = "bar.foo";
  vector<string> expected;
  expected.push_back(key1);
  expected.push_back(key2);
  sort(expected.begin(), expected.end());

  essentia::Pool p;
  p.add("foo.bar", (Real)20.08);
  p.add("bar.foo", (Real)20.09);

  vector<string> result = p.descriptorNames();
  sort(result.begin(), result.end());
  EXPECT_VEC_EQ(result, expected);
}

TEST(Pool, IntegrityCheck) {
  essentia::Pool p;
  p.add("foo.bar", (Real)1.23456789);
  ASSERT_THROW(p.add("foo.bar", "mixed up the types!"), EssentiaException);
}
