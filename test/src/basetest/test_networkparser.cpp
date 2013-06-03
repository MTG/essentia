#include "essentia_gtest.h"
#include "networkparser.h"
using namespace std;
using namespace essentia;
using namespace essentia::scheduler;


TEST(NetworkParser, SimpleBox) {
  const char* network[] = {
    "+------+",
    "| Abcd |",
    "+------+"
  };

  vector<AsciiBox> boxes = AsciiBox::findBoxes(AsciiCanvas(network));

  ASSERT_EQ((size_t)1, boxes.size());
  EXPECT_EQ("Abcd", boxes[0].title);
  EXPECT_EQ(0, boxes[0].posX);
  EXPECT_EQ(0, boxes[0].posY);
  EXPECT_EQ(6, boxes[0].width);
  EXPECT_EQ(1, boxes[0].height);
}


TEST(NetworkParser, BrokenBox) {
  const char* network[] = {
    "+------",
    "| Abcd |",
    "+------+"
  };

  vector<AsciiBox> boxes = AsciiBox::findBoxes(AsciiCanvas(network));

  ASSERT_EQ((size_t)0, boxes.size());
}


TEST(NetworkParser, BrokenBox2) {
  const char* network[] = {
    "+------+",
    "| Abcd |",
    " ------+"
  };

  vector<AsciiBox> boxes = AsciiBox::findBoxes(AsciiCanvas(network));

  ASSERT_EQ((size_t)0, boxes.size());
}



TEST(NetworkParser, MultiBoxes) {
  const char* network[] = {
    "+------+           ",
    "| Abcd |     +------+    ",
    "+------+     |      |  ",
    "             |  Gl  |      ",
    "  +-----+    |  OO  |      ",
    "  | ^_^ |    |  up  |      ",
    "  +-----+    |      |      ",
    "             +------+      ",
    "                           ",
    "                           ",
    "                           "
  };

  vector<AsciiBox> boxes = AsciiBox::findBoxes(AsciiCanvas(network));

  ASSERT_EQ((size_t)3, boxes.size());

  EXPECT_EQ("Abcd", boxes[0].title);
  EXPECT_EQ(0, boxes[0].posX);
  EXPECT_EQ(0, boxes[0].posY);
  EXPECT_EQ(6, boxes[0].width);
  EXPECT_EQ(1, boxes[0].height);

  EXPECT_EQ("", boxes[1].title);
  EXPECT_EQ(13, boxes[1].posX);
  EXPECT_EQ(1,  boxes[1].posY);
  EXPECT_EQ(6,  boxes[1].width);
  EXPECT_EQ(5,  boxes[1].height);

  EXPECT_EQ("^_^", boxes[2].title);
  EXPECT_EQ(2, boxes[2].posX);
  EXPECT_EQ(4, boxes[2].posY);
  EXPECT_EQ(5, boxes[2].width);
  EXPECT_EQ(1, boxes[2].height);

};

TEST(NetworkParser, MultiBoxesWithNoise) {
  const char* network[] = {
    "+------+              ",
    "| Abcd |     +------+    ",
    "+------+ ++  |  Yo  |  ",
    "         ++  |  Gl  |  +-- ",
    "  +-----+    |  OO  |      ",
    "  | ^_^ |    |  up  |   +  ",
    "  +-----+    |      |   |  ",
    "             +------+   |  ",
    "",
    "   ----/\\--ikjg            ",
    "                           "
  };

  vector<AsciiBox> boxes = AsciiBox::findBoxes(AsciiCanvas(network));

  ASSERT_EQ((size_t)4, boxes.size());

  EXPECT_EQ("Abcd", boxes[0].title);
  EXPECT_EQ(0, boxes[0].posX);
  EXPECT_EQ(0, boxes[0].posY);
  EXPECT_EQ(6, boxes[0].width);
  EXPECT_EQ(1, boxes[0].height);

  EXPECT_EQ("Yo", boxes[1].title);
  EXPECT_EQ(13, boxes[1].posX);
  EXPECT_EQ(1,  boxes[1].posY);
  EXPECT_EQ(6,  boxes[1].width);
  EXPECT_EQ(5,  boxes[1].height);

  EXPECT_EQ("", boxes[2].title);
  EXPECT_EQ(9,  boxes[2].posX);
  EXPECT_EQ(2,  boxes[2].posY);
  EXPECT_EQ(0,  boxes[2].width);
  EXPECT_EQ(0,  boxes[2].height);

  EXPECT_EQ("^_^", boxes[3].title);
  EXPECT_EQ(2, boxes[3].posX);
  EXPECT_EQ(4, boxes[3].posY);
  EXPECT_EQ(5, boxes[3].width);
  EXPECT_EQ(1, boxes[3].height);
}


TEST(NetworkParser, Connections) {
  const char* network[] = {
    "+-------------+  +----------+  +------+",
    "| AudioLoader |--| Spectrum |--| MFCC |",
    "+-------------+  +----------+  +------+",
  };

  AsciiDAGParser np(network);

  const char* algos[] = { "AudioLoader", "MFCC", "Spectrum" }; // alphabetically sorted
  EXPECT_VEC_EQ(arrayToVector<string>(algos), np.nodes());

  const vector<pair<string, string> >& c = np.namedEdges();
  ASSERT_EQ((size_t)2, c.size());
  EXPECT_EQ(STRPAIR("AudioLoader", "Spectrum"), c[0]);
  EXPECT_EQ(STRPAIR("Spectrum", "MFCC"),        c[1]);
}

TEST(NetworkParser, ConnectionsAndInstantiation) {
  const char* network[] = {
    "+-------------+  +-------------+  +----------+  +------+",
    "|  MonoLoader |--| FrameCutter |--| Spectrum |--| MFCC |",
    "+-------------+  +-------------+  +----------+  +------+",
  };

  NetworkParser np(network);

  const char* algos[] = { "FrameCutter", "MFCC", "MonoLoader", "Spectrum" }; // alphabetically sorted
  EXPECT_VEC_EQ(arrayToVector<string>(algos), np.algorithms());

  const vector<pair<string, string> >& c = np.namedConnections();
  ASSERT_EQ((size_t)3, c.size());
  EXPECT_EQ(STRPAIR("FrameCutter", "Spectrum"),   c[0]);
  EXPECT_EQ(STRPAIR("MonoLoader", "FrameCutter"), c[1]);
  EXPECT_EQ(STRPAIR("Spectrum", "MFCC"),          c[2]);
}


TEST(NetworkParser, FunkyConnections) {
  const char* network[] = {
    "         +----------+          ",
    "     +---| Spectrum |--+       ",
    "     |   +----------+  |       ",
    "     +-----------+     |       ",
    "+-------------+  |     |       ",
    "| AudioLoader |--+     |       ",
    "+-------------+     +--+       ",
    "      +-------------+          ",
    "      |           +------+     ",
    "      |    +------| MFCC |     ",
    "      |    |      +------+     ",
    "      +----+                   "
  };

  AsciiDAGParser np(network);

  const char* algos[] = { "AudioLoader", "MFCC", "Spectrum" }; // alphabetically sorted
  EXPECT_VEC_EQ(arrayToVector<string>(algos), np.nodes());

  const vector<pair<string, string> >& c = np.namedEdges();
  ASSERT_EQ((size_t)2, c.size());
  EXPECT_EQ(STRPAIR("AudioLoader", "Spectrum"), c[0]);
  EXPECT_EQ(STRPAIR("Spectrum", "MFCC"),        c[1]);
}


TEST(NetworkParser, MultipleConnections) {
  const char* network[] = {
    "                                                                         ",
    "                                  +------+                               ",
    "                               +--| MFCC |                               ",
    "                               |  +------+                               ",
    "+-------------+  +----------+  |                                         ",
    "| AudioLoader |--| Spectrum |--+                                         ",
    "+-------------+  +----------+  |                                         ",
    "                               |                                         ",
    "                               | +-------------------+                   ",
    "                               +-| SpectralCentroid  |                   ",
    "                               | +-------------------+                   ",
    "                               |                                         ",
    "                               |                    +-----+              ",
    "                               |               +----| Key |              ",
    "                               | +------+      |    +-----+              ",
    "                               +-| HPCP |------+                         ",
    "                                 +------+      |    +------+             ",
    "                                               +----| Mode |             ",
    "                                                    +------+             ",
    "                                                                         ",
  };

  AsciiDAGParser np(network);

  const char* algos[] = { "AudioLoader", "HPCP", "Key", "MFCC", "Mode", "SpectralCentroid", "Spectrum" };
  EXPECT_VEC_EQ(arrayToVector<string>(algos), np.nodes());

  const vector<pair<string, string> >& c = np.namedEdges();
  ASSERT_EQ((size_t)6, c.size());
  EXPECT_EQ(STRPAIR("AudioLoader", "Spectrum"), c[0]);
  EXPECT_EQ(STRPAIR("HPCP", "Key"),  c[1]);
  EXPECT_EQ(STRPAIR("HPCP", "Mode"), c[2]);
  EXPECT_EQ(STRPAIR("Spectrum", "HPCP"), c[3]);
  EXPECT_EQ(STRPAIR("Spectrum", "MFCC"), c[4]);
  EXPECT_EQ(STRPAIR("Spectrum", "SpectralCentroid"), c[5]);
}

TEST(NetworkParser, InfiniteSplit) {
  const char* network[] = {
    "+-------------+  +----------+  +------+",
    "| AudioLoader |--| Spectrum |--+ Oops |",
    "+-------------+  +----------+  +------+"
  };

  AsciiDAGParser np(network);

  const char* algos[] = { "AudioLoader", "Spectrum" };
  EXPECT_VEC_EQ(arrayToVector<string>(algos), np.nodes());

  const vector<pair<string, string> >& c = np.namedEdges();
  ASSERT_EQ((size_t)1, c.size());
  EXPECT_EQ(STRPAIR("AudioLoader", "Spectrum"), c[0]);
}

TEST(NetworkParser, BrokenBBox) {
  // The B box at the bottom is broken and might lead to infinite paths
  // This is the real-life case that lead to the previous isolated unittest
  const char* network[] = {
    "            +---+               ",
    "         +--| B |--+            ",
    "         |  +---+  |            ",
    "   +---+ |         |  +---+     ",
    "   | A |-+         +--| C |     ",
    "   +---+ |         |  +---+     ",
    "         |  +---+  |            ",
    "         +--| B +--+            ",
    "            +---+ __            ",
    "                 |\\            ",
    "                 ' \\ This box is broken "
  };

  AsciiDAGParser np(network);

}
