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
#include "peak.h"
using namespace std;
using namespace essentia;
using util::peaksToReals;
using util::realsToPeaks;
using util::ComparePeakMagnitude;
using util::ComparePeakPosition;


TEST(Peak, Constructor) {
  util::Peak p1;
  util::Peak p2(1, 2);
  util::Peak p3(1, 2.52);
  util::Peak p4(1.52, 2);
  util::Peak p5(1.52, 2.52);

  util::Peak p6(pair<int, int>(1, 2));
  util::Peak p7(pair<int, Real>(1, 2.52));
  util::Peak p8(pair<Real, int>(1.52, 2));
  util::Peak p9(pair<Real, Real>(1.52, 2.52));

  util::Peak p10(p2);

  EXPECT_EQ(p1.position, 0);
  EXPECT_EQ(p1.magnitude, 0);

  EXPECT_EQ(p2.position, 1);
  EXPECT_EQ(p2.magnitude, 2);

  EXPECT_EQ(p3.position, 1);
  EXPECT_EQ(p3.magnitude, Real(2.52));

  EXPECT_EQ(p4.position, Real(1.52));
  EXPECT_EQ(p4.magnitude, 2);

  EXPECT_EQ(p5.position, Real(1.52));
  EXPECT_EQ(p5.magnitude, Real(2.52));

  EXPECT_EQ(p6.position, 1);
  EXPECT_EQ(p6.magnitude, 2);

  EXPECT_EQ(p7.position, 1);
  EXPECT_EQ(p7.magnitude, Real(2.52));

  EXPECT_EQ(p8.position, Real(1.52));
  EXPECT_EQ(p8.magnitude, 2);

  EXPECT_EQ(p9.position, Real(1.52));
  EXPECT_EQ(p9.magnitude, Real(2.52));

  EXPECT_EQ(p10.position, 1);
  EXPECT_EQ(p10.magnitude, 2);
}

TEST(Peak, Assignment) {
  util::Peak p;
  p = util::Peak(1,2);
  EXPECT_EQ(p.position, 1);
  EXPECT_EQ(p.magnitude, 2);

  p = std::pair<Real, int>(Real(2.4), 5);
  EXPECT_EQ(p.position, Real(2.4));
  EXPECT_EQ(p.magnitude, 5);
}

TEST(Peak, Equality) {
  EXPECT_EQ(util::Peak(12.567, 15.789), util::Peak(12.567, 15.789));
  EXPECT_NE(util::Peak(12, 15.789), util::Peak(12.567, 15.789));
  EXPECT_NE(util::Peak(12.567, 15.9), util::Peak(12.567, 15.789));
}

TEST(Peak, Greater) {
  EXPECT_TRUE(util::Peak(14.567, 15.789) > util::Peak(12.567, 14.789));
  EXPECT_TRUE(util::Peak(12.567, 15.789) > util::Peak(12.567, 14.789));
  EXPECT_TRUE(util::Peak(2.567, 15.789)  > util::Peak(12.567, 14.789));

  EXPECT_TRUE(util::Peak(14.567, 15.789) >= util::Peak(12.567, 14.789));
  EXPECT_TRUE(util::Peak(12.567, 15.789) >= util::Peak(12.567, 14.789));
  EXPECT_TRUE(util::Peak(2.567, 15.789)  >= util::Peak(12.567, 14.789));
  EXPECT_TRUE(util::Peak(12.567, 15.789) >= util::Peak(12.567, 15.789));
}

TEST(Peak, Less) {
  EXPECT_TRUE(util::Peak(12.567, 14.789) < util::Peak(14.567, 15.789));
  EXPECT_TRUE(util::Peak(14.567, 14.789) < util::Peak(12.567, 15.789));
  EXPECT_TRUE(util::Peak(2.567, 14.789)  < util::Peak(2.567, 15.789));

  EXPECT_TRUE(util::Peak(12.567, 14.789) <= util::Peak(14.567, 15.789));
  EXPECT_TRUE(util::Peak(12.567, 14.789) <= util::Peak(12.567, 15.789));
  EXPECT_TRUE(util::Peak(12.567, 14.789) <= util::Peak(2.567, 15.789));
  EXPECT_TRUE(util::Peak(12.567, 15.789) <= util::Peak(12.567, 15.789));
}

TEST(Peak, PeaksToReals) {
  Real positionsArray [] = {1, 0, 1, 4, 5};
  Real magnitudesArray [] = {2, 2, 3, 1, 6};
  util::Peak peaksArray [] = {util::Peak(1,2),util::Peak(0,2),util::Peak(1,3),util::Peak(4,1),util::Peak(5,6)};
  vector<util::Peak> peaks = arrayToVector<util::Peak>(peaksArray);
  vector<Real> positions, magnitudes;
  peaksToReals(peaks, positions, magnitudes);
  EXPECT_VEC_EQ(positions, arrayToVector<Real>(positionsArray));
  EXPECT_VEC_EQ(magnitudes, arrayToVector<Real>(magnitudesArray));
}

TEST(Peak, RealsToPeaks) {
  Real positionsArray [] = {1, 0, 1, 4, 5};
  Real magnitudesArray [] = {2, 2, 3, 1, 6};
  util::Peak peaksArray [] = {util::Peak(1,2),util::Peak(0,2),util::Peak(1,3),util::Peak(4,1),util::Peak(5,6)};
  vector<Real> positions = arrayToVector<Real>(positionsArray);
  vector<Real> magnitudes = arrayToVector<Real>(magnitudesArray);
  vector<util::Peak> peaks = realsToPeaks(positions, magnitudes);
  EXPECT_VEC_EQ(peaks, arrayToVector<util::Peak>(peaksArray));
}

TEST(Peak, SortMagnitudeDescendingPositionDescending) {
  Real positions [] = {1, 0, 1, 4, 5};
  Real magnitudes[] = {2, 2, 3, 1, 6};
  vector<util::Peak> peaks = realsToPeaks(arrayToVector<Real>(positions),
                                          arrayToVector<Real>(magnitudes));
  sort(peaks.begin(), peaks.end(),
       ComparePeakMagnitude<std::greater<Real>, std::greater<Real> >());
  Real expectedPos[] = {5,1,1,0,4};
  Real expectedMag[] = {6,3,2,2,1};
  vector<util::Peak> expectedPeaks= realsToPeaks(arrayToVector<Real>(expectedPos),
                                                 arrayToVector<Real>(expectedMag));
  EXPECT_VEC_EQ(peaks, expectedPeaks);
}

TEST(Peak, SortMagnitudeAscendingPositionDescending) {
  Real positions [] = {1, 0, 1, 4, 5};
  Real magnitudes[] = {2, 2, 3, 1, 6};
  vector<util::Peak> peaks = realsToPeaks(arrayToVector<Real>(positions),
                                          arrayToVector<Real>(magnitudes));
  sort(peaks.begin(), peaks.end(),
       ComparePeakMagnitude<std::less<Real>, std::greater<Real> >());
  Real expectedPos[] = {4,1,0,1,5};
  Real expectedMag[] = {1,2,2,3,6};
  vector<util::Peak> expectedPeaks= realsToPeaks(arrayToVector<Real>(expectedPos),
                                                 arrayToVector<Real>(expectedMag));
  EXPECT_VEC_EQ(peaks, expectedPeaks);
}

TEST(Peak, SortMagnitudeDescendingPositionAscending) {
  Real positions [] = {1, 0, 1, 4, 5};
  Real magnitudes[] = {2, 2, 3, 1, 6};
  vector<util::Peak> peaks = realsToPeaks(arrayToVector<Real>(positions),
                                          arrayToVector<Real>(magnitudes));
  sort(peaks.begin(), peaks.end(),
       ComparePeakMagnitude<std::greater<Real>, std::less<Real> >());
  Real expectedPos[] = {5,1,0,1,4};
  Real expectedMag[] = {6,3,2,2,1};
  vector<util::Peak> expectedPeaks= realsToPeaks(arrayToVector<Real>(expectedPos),
                                                 arrayToVector<Real>(expectedMag));
  EXPECT_VEC_EQ(peaks, expectedPeaks);
}

TEST(Peak, SortMagnitudeAscendingPositionAscending) {
  Real positions [] = {1, 0, 1, 4, 5};
  Real magnitudes[] = {2, 2, 3, 1, 6};
  vector<util::Peak> peaks = realsToPeaks(arrayToVector<Real>(positions),
                                          arrayToVector<Real>(magnitudes));
  sort(peaks.begin(), peaks.end(),
       ComparePeakMagnitude<std::less<Real>, std::less<Real> >());
  Real expectedPos[] = {4,0,1,1,5};
  Real expectedMag[] = {1,2,2,3,6};
  vector<util::Peak> expectedPeaks= realsToPeaks(arrayToVector<Real>(expectedPos),
                                                 arrayToVector<Real>(expectedMag));
  EXPECT_VEC_EQ(peaks, expectedPeaks);
}


////************************************

TEST(Peak, SortPositionDescendingMagnitudeDescending) {
  Real positions [] = {1, 0, 1, 4, 5};
  Real magnitudes[] = {2, 2, 3, 1, 6};
  vector<util::Peak> peaks = realsToPeaks(arrayToVector<Real>(positions),
                                          arrayToVector<Real>(magnitudes));
  sort(peaks.begin(), peaks.end(),
       ComparePeakPosition<std::greater<Real>, std::greater<Real> >());
  Real expectedPos[] = {5,4,1,1,0};
  Real expectedMag[] = {6,1,3,2,2};
  vector<util::Peak> expectedPeaks= realsToPeaks(arrayToVector<Real>(expectedPos),
                                                 arrayToVector<Real>(expectedMag));
  EXPECT_VEC_EQ(peaks, expectedPeaks);
}

TEST(Peak, SortPositionAscendingMagnitudeDescending) {
  Real positions [] = {1, 0, 1, 4, 5};
  Real magnitudes[] = {2, 2, 3, 1, 6};
  vector<util::Peak> peaks = realsToPeaks(arrayToVector<Real>(positions),
                                          arrayToVector<Real>(magnitudes));
  sort(peaks.begin(), peaks.end(),
       ComparePeakPosition<std::less<Real>, std::greater<Real> >());
  Real expectedPos[] = {0,1,1,4,5};
  Real expectedMag[] = {2,3,2,1,6};
  vector<util::Peak> expectedPeaks= realsToPeaks(arrayToVector<Real>(expectedPos),
                                                 arrayToVector<Real>(expectedMag));
  EXPECT_VEC_EQ(peaks, expectedPeaks);
}

TEST(Peak, SortPositionDescendingMagnitudeAscending) {
  Real positions [] = {1, 0, 1, 4, 5};
  Real magnitudes[] = {2, 2, 3, 1, 6};
  vector<util::Peak> peaks = realsToPeaks(arrayToVector<Real>(positions),
                                          arrayToVector<Real>(magnitudes));
  sort(peaks.begin(), peaks.end(),
       ComparePeakPosition<std::greater<Real>, std::less<Real> >());
  Real expectedPos[] = {5,4,1,1,0};
  Real expectedMag[] = {6,1,2,3,2};
  vector<util::Peak> expectedPeaks= realsToPeaks(arrayToVector<Real>(expectedPos),
                                                 arrayToVector<Real>(expectedMag));
  EXPECT_VEC_EQ(peaks, expectedPeaks);
}

TEST(Peak, SortPositionAscendingMagnitudeAscending) {
  Real positions [] = {1, 0, 1, 4, 5};
  Real magnitudes[] = {2, 2, 3, 1, 6};
  vector<util::Peak> peaks = realsToPeaks(arrayToVector<Real>(positions),
                                          arrayToVector<Real>(magnitudes));
  sort(peaks.begin(), peaks.end(),
       ComparePeakPosition<std::less<Real>, std::less<Real> >());
  Real expectedPos[] = {0,1,1,4,5};
  Real expectedMag[] = {2,2,3,1,6};
  vector<util::Peak> expectedPeaks= realsToPeaks(arrayToVector<Real>(expectedPos),
                                                 arrayToVector<Real>(expectedMag));
  EXPECT_VEC_EQ(peaks, expectedPeaks);
}
