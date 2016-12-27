/*
 * Copyright (C) 2006-2016  Music Technology Group - Universitat Pompeu Fabra
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
#include "scheduler/network.h"
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
                                        "filename", "test/audio/recorded/cat_purrrr.wav",
                                        "computeMD5", true);
    essentia::Pool p;

    loader->output("audio")           >>  PC(p, "audio");
    loader->output("sampleRate")      >>  PC(p, "samplerate");
    loader->output("numberChannels")  >>  PC(p, "channels");
    loader->output("md5")             >>  PC(p, "md5");
    loader->output("codec")           >>  PC(p, "codec");
    loader->output("bit_rate")        >>  PC(p, "bit_rate");

    Network(loader).run();

    EXPECT_EQ(44100,   p.value<Real>("samplerate"));
    EXPECT_EQ(2,       p.value<Real>("channels"));
    EXPECT_EQ(219343,  (int)p.value<vector<StereoSample> >("audio").size());
    EXPECT_EQ("426fe5cf5ac3730f8c8db2a760e2b819", p.value<string>("md5"));
    EXPECT_EQ("pcm_s16le", p.value<string>("codec"));
    EXPECT_EQ(1411200, p.value<Real>("bit_rate"));
}

TEST(AudioLoader, SampleFormatConversion) {
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    Algorithm* loader24 =  factory.create("AudioLoader",
                                          "filename", "test/audio/recorded/cat_purrrr24bit.wav",
                                          "computeMD5", true);
    essentia::Pool p;

    loader24->output("audio")           >>  PC(p, "audio24");
    loader24->output("sampleRate")      >>  PC(p, "samplerate24");
    loader24->output("numberChannels")  >>  PC(p, "channels24");
    loader24->output("md5")             >>  PC(p, "md5");
    loader24->output("codec")           >>  PC(p, "codec");
    loader24->output("bit_rate")        >>  PC(p, "bit_rate");

    Network(loader24).run();

    EXPECT_EQ(44100,   p.value<Real>("samplerate24"));
    EXPECT_EQ(2,       p.value<Real>("channels24"));
    EXPECT_EQ(219343,  (int)p.value<vector<StereoSample> >("audio24").size());
    EXPECT_EQ("0616e4672c8a616a21e4873e39da1739", p.value<string>("md5"));
    EXPECT_EQ("pcm_s24le", p.value<string>("codec"));
    EXPECT_EQ(2116800, p.value<Real>("bit_rate"));


    // FIXME: the following should work
    //loader->configure("filename", "test/audio/recorded/britney32bit.wav");
    //p.clear();

    Algorithm* loader32 =  factory.create("AudioLoader",
                                          "filename", "test/audio/recorded/cat_purrrr32bit.wav",
                                          "computeMD5", true);

    loader32->output("audio")           >>  PC(p, "audio32");
    loader32->output("sampleRate")      >>  PC(p, "samplerate32");
    loader32->output("numberChannels")  >>  PC(p, "channels32");
    loader32->output("md5")             >>  PC(p, "md5");
    loader32->output("codec")           >>  PC(p, "codec");
    loader32->output("bit_rate")        >>  PC(p, "bit_rate");

    Network(loader32).run();

    EXPECT_EQ(44100,   p.value<Real>("samplerate32"));
    EXPECT_EQ(2,       p.value<Real>("channels32"));
    EXPECT_EQ(219343,  (int)p.value<vector<StereoSample> >("audio32").size());
    EXPECT_EQ("622be297e9e7f75f6fdf6c571999d1ca", p.value<string>("md5"));
    EXPECT_EQ("pcm_s32le", p.value<string>("codec"));
    EXPECT_EQ(2822400, p.value<Real>("bit_rate"));
}
