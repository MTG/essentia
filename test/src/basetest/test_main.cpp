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
#include "algorithmfactory.h"
#include "customalgos.h"

static const bool FULL_DEBUG = false;

int main(int argc, char **argv) {
  ::essentia::init();

  if (FULL_DEBUG) {
    ::essentia::setDebugLevel(::essentia::EAll);
    ::essentia::unsetDebugLevel(::essentia::EMemory);
  }

  REGISTER_ALGO(CompositeAlgo);
  REGISTER_ALGO(DiamondShapeAlgo);
  REGISTER_ALGO(CopyAlgo);
  REGISTER_ALGO(TeeAlgo);
  REGISTER_ALGO(TeeProxyAlgo);

  REGISTER_ALGO(A);
  REGISTER_ALGO(B);
  REGISTER_ALGO(C);
  REGISTER_ALGO(D);
  REGISTER_ALGO(E);
  REGISTER_ALGO(F);
  REGISTER_ALGO(G);
  REGISTER_ALGO(H);
  REGISTER_ALGO(I);
  REGISTER_ALGO(J);

  REGISTER_ALGO(B1);
  REGISTER_ALGO(E1);
  REGISTER_ALGO(F1);
  REGISTER_ALGO(D1);
  REGISTER_ALGO(A1);
  REGISTER_ALGO(A2);
  REGISTER_ALGO(A3);
  REGISTER_ALGO(A4);
  REGISTER_ALGO(A5);
  REGISTER_ALGO(A6);

  REGISTER_ALGO(DevNullSample);
  REGISTER_ALGO(PoolStorageFrame);

  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();

  essentia::shutdown();

  return result;
}

DEFINE_TEST_ALGO(CompositeAlgo);
DEFINE_TEST_ALGO(DiamondShapeAlgo);
DEFINE_TEST_ALGO(CopyAlgo);
DEFINE_TEST_ALGO(TeeAlgo);
DEFINE_TEST_ALGO(TeeProxyAlgo);

DEFINE_TEST_ALGO(A);
DEFINE_TEST_ALGO(B);
DEFINE_TEST_ALGO(C);
DEFINE_TEST_ALGO(D);
DEFINE_TEST_ALGO(E);
DEFINE_TEST_ALGO(F);
DEFINE_TEST_ALGO(G);
DEFINE_TEST_ALGO(H);
DEFINE_TEST_ALGO(I);
DEFINE_TEST_ALGO(J);

DEFINE_TEST_ALGO(B1);
DEFINE_TEST_ALGO(E1);
DEFINE_TEST_ALGO(F1);
DEFINE_TEST_ALGO(D1);
DEFINE_TEST_ALGO(A1);
DEFINE_TEST_ALGO(A2);
DEFINE_TEST_ALGO(A3);
DEFINE_TEST_ALGO(A4);
DEFINE_TEST_ALGO(A5);
DEFINE_TEST_ALGO(A6);

const char* essentia::streaming::DevNullSample::name = "DevNull";
const char* essentia::streaming::DevNullSample::version = "1.0";
const char* essentia::streaming::DevNullSample::description = "";

const char* essentia::streaming::PoolStorageFrame::name = "PoolStorage";
const char* essentia::streaming::PoolStorageFrame::version = "1.0";
const char* essentia::streaming::PoolStorageFrame::description = "";
