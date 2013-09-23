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
                                        "filename", "test/audio/recorded/cat_purrrr.wav");
    essentia::Pool p;

    loader->output("audio")           >>  PC(p, "audio");
    loader->output("sampleRate")      >>  PC(p, "samplerate");
    loader->output("numberChannels")  >>  PC(p, "channels");

    Network(loader).run();

    EXPECT_EQ(44100,   p.value<Real>("samplerate"));
    EXPECT_EQ(2,       p.value<Real>("channels"));
    EXPECT_EQ(219343,  (int)p.value<vector<StereoSample> >("audio").size());
}

TEST(AudioLoader, SampleFormatConversion) {
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    Algorithm* loader24 =  factory.create("AudioLoader",
                                          "filename", "test/audio/recorded/cat_purrrr24bit.wav");
    essentia::Pool p;

    loader24->output("audio")           >>  PC(p, "audio24");
    loader24->output("sampleRate")      >>  PC(p, "samplerate24");
    loader24->output("numberChannels")  >>  PC(p, "channels24");

    Network(loader24).run();

    EXPECT_EQ(44100,   p.value<Real>("samplerate24"));
    EXPECT_EQ(2,       p.value<Real>("channels24"));
    EXPECT_EQ(219343,  (int)p.value<vector<StereoSample> >("audio24").size());

    // FIXME: the following should work
    //loader->configure("filename", "test/audio/recorded/britney32bit.wav");
    //p.clear();

    Algorithm* loader32 =  factory.create("AudioLoader",
                                          "filename", "test/audio/recorded/cat_purrrr32bit.wav");

    loader32->output("audio")           >>  PC(p, "audio32");
    loader32->output("sampleRate")      >>  PC(p, "samplerate32");
    loader32->output("numberChannels")  >>  PC(p, "channels32");

    Network(loader32).run();

    EXPECT_EQ(44100,   p.value<Real>("samplerate32"));
    EXPECT_EQ(2,       p.value<Real>("channels32"));
    EXPECT_EQ(219343,  (int)p.value<vector<StereoSample> >("audio32").size());
}
