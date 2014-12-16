/*
 * Copyright (C) 2006-2014  Music Technology Group - Universitat Pompeu Fabra
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
#include "essentiamath.h"
using namespace std;
using namespace essentia;


TEST(Math, NextPowerOfTwo) {
  EXPECT_EQ(8, nextPowerTwo(6));
  EXPECT_EQ(8, nextPowerTwo(7));
  EXPECT_EQ(8, nextPowerTwo(8));
  EXPECT_EQ(16, nextPowerTwo(9));

  // test 64 bit
  // n = 1 << 40, but we need to do it in 2 steps because we get a shift
  //              count overflow otherwise
  long long int n = (1 << 20);
  n <<= 20;
  EXPECT_EQ(n, nextPowerTwo(n-230875));
  EXPECT_EQ(n, nextPowerTwo(n-1));
  EXPECT_EQ(n, nextPowerTwo(n));
  EXPECT_EQ(2*n, nextPowerTwo(n+1));

}
