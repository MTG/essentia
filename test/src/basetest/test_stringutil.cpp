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
using namespace essentia;


TEST(StringUtil, TokenizeSimple) {
  string str = "hello\nthis\nis\na\ntest";

  vector<string> tokens = tokenize(str, "\n");
  const char* expected[] = { "hello", "this", "is", "a", "test" };

  EXPECT_VEC_EQ(tokens, arrayToVector<string>(expected));
}

TEST(StringUtil, TokenizeEmpty) {
  vector<string> tokens = tokenize("", "\n");
  EXPECT_TRUE(tokens.empty());
}


TEST(StringUtil, Strip) {
  EXPECT_EQ(strip("  \t To infinity and beyond!  \n"),
            "To infinity and beyond!");
}

TEST(StringUtil, Lower) {
  EXPECT_EQ(toLower(""), "");
  EXPECT_EQ(toLower("ABC123"), "abc123");
  EXPECT_EQ(toLower("l33t HAXX0rz"), "l33t haxx0rz");
}

TEST(StringUtil, Upper) {
  EXPECT_EQ(toUpper(""), "");
  EXPECT_EQ(toUpper("AbC123"), "ABC123");
  EXPECT_EQ(toUpper("l33t HAXX0rz"), "L33T HAXX0RZ");
}
