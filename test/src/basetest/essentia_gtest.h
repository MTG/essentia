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

#ifndef ESSENTIA_GTEST_H
#define ESSENTIA_GTEST_H

#include <iostream>
#include <gtest/gtest.h>
#include "essentia.h"
#include "algorithmfactory.h"
#include "streamingalgorithm.h"
#include "streamingalgorithmcomposite.h"
#include "poolstorage.h"

#define DBG(x) E_DEBUG(EUnittest, x)

#define EXPECT_VEC_EQ(x, y) {                                                         \
  ASSERT_EQ(x.size(), y.size()) << "Vectors " #x " and " #y " are of unequal length"; \
                                                                                      \
  for (int i=0; i<(int)x.size(); i++) {                                               \
    EXPECT_EQ(x[i], y[i]) << "Vectors " #x " and " #y " differ at index " << i;       \
  }                                                                                   \
}

#define EXPECT_MATRIX_EQ(x, y) {                                                                                         \
  ASSERT_EQ(x.size(), y.size()) << "Matrices " #x " and " #y " don't have the same number of rows";                      \
                                                                                                                         \
  for (int i=0; i<(int)x.size(); i++) {                                                                                  \
    ASSERT_EQ(x[i].size(), y[i].size()) << "Row " << i << " doesn't have the same number of columns for " #x " and " #y; \
                                                                                                                         \
    for (int j=0; j<(int)x[i].size(); j++) {                                                                             \
      EXPECT_EQ(x[i][j], y[i][j]) << "Matrix " #x " and " #y " differ at position (" << i << ", " << j << ")";           \
    }                                                                                                                    \
  }                                                                                                                      \
}

#define STRPAIR(s1, s2) make_pair(string(s1), string(s2))

#define VISIBLE_NETWORK(nw) NetworkParser(nw, false).network()->visibleNetworkRoot()


#endif // ESSENTIA_GTEST_H
