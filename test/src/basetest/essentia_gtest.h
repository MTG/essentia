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
