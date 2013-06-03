#include "essentia_gtest.h"
#include "../../../src/scheduler/network.h"
#include "streamingalgorithmcomposite.h"
#include "vectorinput.h"
#include "vectoroutput.h"
#include "customalgos.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


TEST(AudioLoader, SimpleLoad) {
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    Algorithm* loader =  factory.create("AudioLoader",
                                        "filename", "test/audio/recorded/britney.wav");
    essentia::Pool p;

    loader->output("audio")           >>  PC(p, "audio");
    loader->output("sampleRate")      >>  PC(p, "samplerate");
    loader->output("numberChannels")  >>  PC(p, "channels");

    Network(loader).run();

    EXPECT_EQ(44100,   p.value<Real>("samplerate"));
    EXPECT_EQ(2,       p.value<Real>("channels"));
    EXPECT_EQ(1323588, p.value<vector<StereoSample> >("audio").size());
}

TEST(AudioLoader, SampleFormatConversion) {
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    Algorithm* loader24 =  factory.create("AudioLoader",
                                          "filename", "test/audio/recorded/britney24bit.wav");
    essentia::Pool p;

    loader24->output("audio")           >>  PC(p, "audio24");
    loader24->output("sampleRate")      >>  PC(p, "samplerate24");
    loader24->output("numberChannels")  >>  PC(p, "channels24");

    Network(loader24).run();

    EXPECT_EQ(44100,   p.value<Real>("samplerate24"));
    EXPECT_EQ(2,       p.value<Real>("channels24"));
    EXPECT_EQ(1323588, p.value<vector<StereoSample> >("audio24").size());

    // FIXME: the following should work
    //loader->configure("filename", "test/audio/recorded/britney32bit.wav");
    //p.clear();

    Algorithm* loader32 =  factory.create("AudioLoader",
                                          "filename", "test/audio/recorded/britney32bit.wav");

    loader32->output("audio")           >>  PC(p, "audio32");
    loader32->output("sampleRate")      >>  PC(p, "samplerate32");
    loader32->output("numberChannels")  >>  PC(p, "channels32");

    Network(loader32).run();

    EXPECT_EQ(44100,   p.value<Real>("samplerate32"));
    EXPECT_EQ(2,       p.value<Real>("channels32"));
    EXPECT_EQ(1323588, p.value<vector<StereoSample> >("audio32").size());
}
