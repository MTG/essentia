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
using namespace std;
using essentia::Real;
using essentia::EssentiaException;

typedef essentia::Parameter Param;

class Parameter : public ::testing::Test {
 protected:
  vector<Real> _vr1, _vr2, _vr3;
  vector<string> _vs1, _vs2, _vs3;

  virtual void SetUp() {
    // _vr1 and _vr2 must be equal
    _vr1.push_back(3.56456);
    _vr1.push_back(5);
    _vr1.push_back(-19);

    _vr2.push_back(3.56456);
    _vr2.push_back(5);
    _vr2.push_back(-19);

    _vr3.push_back(4);
    _vr3.push_back(-78.2);

    // _vs1 and _vs2 must be equal
    _vs1.push_back("foo");
    _vs1.push_back("bar");
    _vs1.push_back("!");

    _vs2.push_back("foo");
    _vs2.push_back("bar");
    _vs2.push_back("!");

    _vs3.push_back("hello");
    _vs3.push_back("world");
  }
};


TEST_F(Parameter, EqualityReal) {
  EXPECT_EQ(Param(405.432), Param(405.432));
  EXPECT_NE(Param(674.5254), Param(76574.3246));
}

TEST_F(Parameter, RealToString) {
  EXPECT_EQ(Param(69.08789).toString().substr(0, 8), "69.08789");
}

TEST_F(Parameter, EqualityString) {
  EXPECT_EQ(Param("yo dawg"), Param("yo dawg"));
  EXPECT_NE(Param("foobar"), Param("barfoo"));
}

TEST_F(Parameter, StringToString) {
  EXPECT_EQ(Param("I write a LOT of tests").toString(), "I write a LOT of tests");
}

TEST_F(Parameter, EqualityBool) {
  EXPECT_EQ(Param(false), Param(false));
  EXPECT_NE(Param(true), Param(false));
}

TEST_F(Parameter, BoolToString) {
  EXPECT_EQ(Param(true).toString(), "true");
}

TEST_F(Parameter, EqualityInt) {
  EXPECT_EQ(Param(69), Param(69));
  EXPECT_NE(Param(45), Param(-45));
}

TEST_F(Parameter, IntToString) {
  EXPECT_EQ(Param(-7891).toString(), "-7891");
}

TEST_F(Parameter, ToLower) {
  Param p("I hAtE wHeN pEoPlE dO tHiS cAsE");
  EXPECT_EQ(p.toLower(), "i hate when people do this case");
}

// vector tests
TEST_F(Parameter, EqualityVectorReal) {
  EXPECT_EQ(Param(_vr1), Param(_vr2));
  EXPECT_NE(Param(_vr2), Param(_vr3));
}

TEST_F(Parameter, EqualityVectorString) {
  EXPECT_EQ(Param(_vs1), Param(_vs2));
  EXPECT_NE(Param(_vs2), Param(_vs3));
}

TEST_F(Parameter, VectorStringToString) {
  EXPECT_EQ(Param(_vs1).toString(), "[\"foo\", \"bar\", \"!\"]");
}

// map tests
TEST_F(Parameter, EqualityMapReal) {
  map<string, Real> m1;
  m1["foo"] = (Real)123.456;
  m1["bar"] = (Real)789.123;

  map<string, Real> m2;
  m2["foo"] = (Real)123.456;
  m2["bar"] = (Real)789.123;

  EXPECT_EQ(Param(m1), Param(m2));
}

TEST_F(Parameter, EqualityMapVectorReal) {
  map<string, vector<Real> > m1;
  m1["foo"] = _vr1;
  m1["bar"] = _vr3;

  map<string, vector<Real> > m2;
  m2["foo"] = _vr2; // should be equivalent to m1["foo"]
  m2["bar"] = _vr3;

  EXPECT_EQ(Param(m1), Param(m2));
}

TEST_F(Parameter, MapVectorIntToString) {
  vector<int> v1(3);
  v1[0] = 69;
  v1[1] = -69;
  v1[2] = 101;

  vector<int> v2(3);
  v2[0] = 123;
  v2[1] = 456;
  v2[2] = -789;

  map<string, vector<int> > m1;
  m1["foo"] = v1;
  m1["bar"] = v2;

  // Note: I am sort of cheating in this test because there is no inherent order
  // to the elements in a map.
  EXPECT_EQ(Param(m1).toString(), "{bar: [123, 456, -789], foo: [69, -69, 101]}");
}

TEST_F(Parameter, ToReal) {
  Param p(10.01);
  EXPECT_EQ(p.toReal(), (Real)10.01);
}

TEST_F(Parameter, ToBool) {
  Param p(true);
  EXPECT_EQ(p.toBool(), true);
}

TEST_F(Parameter, ToInt) {
  Param p(-78);
  EXPECT_EQ(p.toInt(), -78);
}

TEST_F(Parameter, ToDouble) {
  Param p(78.456123);
  EXPECT_EQ(p.toDouble(), (Real)78.456123);
}

TEST_F(Parameter, ToFloat) {
  Param p(3.1459);
  EXPECT_EQ(p.toFloat(), (Real)3.1459);
}

TEST_F(Parameter, ToVectorReal) {
  Param p(_vr1);
  EXPECT_VEC_EQ(p.toVectorReal(), _vr1);
}

TEST_F(Parameter, ToMapReal) {
  map<string, Real> m;
  m["foo"] = 123.456;
  m["bar"] = 456.789;
  Param p(m);

  EXPECT_TRUE(p.toMapReal() == m);
}

TEST_F(Parameter, NotConvertible) {
  Param p("this is string!");

  ASSERT_THROW(p.toReal(), EssentiaException);
}

TEST_F(Parameter, ParamToStream) {
  Param strP("str");
  Param intP(100);
  Param realP(10.02);
  Param boolP(true);
  vector<Real> v(10, 1.0);
  Param vecP(v);
  map<string, Real> m;
  m["foo"] = 1.0;
  Param mapP(m);

  ostringstream stStr;
  stStr << strP.type();
  EXPECT_EQ(stStr.str(), "STRING");
  ostringstream stInt;
  stInt << intP.type();
  EXPECT_EQ(stInt.str(), "INT");
  ostringstream stReal;
  stReal << realP.type();
  EXPECT_EQ(stReal.str(), "REAL");
  ostringstream stBool;
  stBool << boolP.type();
  EXPECT_EQ(stBool.str(), "BOOL");
  ostringstream stVector;
  stVector << vecP.type();
  EXPECT_EQ(stVector.str(), "VECTOR_REAL");
  ostringstream stMap;
  stMap << mapP.type();
  EXPECT_EQ(stMap.str(), "MAP_REAL");
}
