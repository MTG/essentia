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
