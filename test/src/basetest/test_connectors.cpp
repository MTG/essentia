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
using namespace std;
using namespace essentia;
using namespace essentia::streaming;
using namespace essentia::scheduler;


TEST(Connectors, SimplePush) {
  DBG("source1");
  Source<float> source("Source1");
  DBG("source1 ok");
  Sink<float> sink("Sink1");
  DBG("sink ok");
  DBG("AVAIL: " << source.available());

  DBG("connect");
  connect(source, sink);
  DBG("connect ok");
  DBG("AVAIL: " << source.available());

  //Vector v; v << 1, 2, 3;

  DBG("push");

  source.push(1.0f);
  source.push(2.0f);
  source.push(3.0f);
  DBG("pop");

  DBG("AVAIL: " << source.available());
  DBG("SINK AVAIL: " << sink.available());

  EXPECT_EQ(sink.pop(), 1);
  DBG("pop1");
  EXPECT_EQ(sink.pop(), 2);
  DBG("pop2");
  EXPECT_EQ(sink.pop(), 3);
  DBG("pop3");
  DBG("SINK AVAIL: " << sink.available());
  DBG((sink.acquire(1) ? "got 1" : "got none"));
  ASSERT_THROW(sink.pop(), EssentiaException);
}

TEST(Connectors, Memory) {
  Source<int>* src = new Source<int>("Source1");
  SourceProxy<int>* proxy = new SourceProxy<int>("Proxy1");
  Sink<int>* sink = new Sink<int>("Sink1");

  *src >> *proxy;
  *proxy >> *sink;

  ASSERT_EQ((size_t)1, src->sinks().size());
  ASSERT_EQ(sink, src->sinks()[0]);

  delete sink;

  ASSERT_EQ((size_t)0, src->sinks().size());

  delete proxy;
  delete src;
}

TEST(Connectors, SourceProxy) {
  Source<string> src("Source1");
  SourceProxy<string> proxy("Proxy1");
  Sink<string> sink("Sink1");

  attach(src, proxy);
  ASSERT_EQ(src.available(), proxy.available());

  ASSERT_THROW(sink.available(), EssentiaException);

  connect(proxy, sink);

  src.push(string("hello"));
  src.push(string("world"));

  ASSERT_EQ(sink.available(), 2);

  EXPECT_EQ(sink.pop(), "hello");
  EXPECT_EQ(sink.pop(), "world");

  ASSERT_EQ(sink.available(), 0);
  ASSERT_THROW(sink.pop(), EssentiaException);

  // try with attach after connect, and disconnect, etc...
}

TEST(Connectors, MultiReaderTest) {
  Source<float> source("Source1");
  Sink<float> sink("Sink1");
  Sink<float> sink2("Sink2");

  DBG("connect");
  connect(source, sink);
  connect(source, sink2);
  DBG("connect ok");

  DBG("push");
  source.push(1.0f);
  source.push(2.0f);
  source.push(3.0f);
  DBG("pop");

  ASSERT_EQ(sink.available(), 3);
  ASSERT_EQ(sink2.available(), 3);
  DBG("SINK AVAIL: " << sink.available());

  EXPECT_EQ(sink.pop(), 1);
  EXPECT_EQ(sink2.pop(), 1);
  EXPECT_EQ(sink2.pop(), 2);
  EXPECT_EQ(sink.pop(), 2);
  EXPECT_EQ(sink2.pop(), 3);
  EXPECT_EQ(sink.pop(), 3);
  ASSERT_THROW(sink.pop(), EssentiaException);
  ASSERT_THROW(sink2.pop(), EssentiaException);
}

TEST(Connectors, SourceProxyConnectBeforeAttach) {
  Source<string> src("Source1");
  SourceProxy<string> proxy("Proxy1");
  Sink<string> sink("Sink1");

  ASSERT_THROW(sink.available(), EssentiaException);
  DBG("connecting");
  // we cannot connect because we would need to have a buffer ready to assign
  // IDs, but we don't have that handy yet...
  //ASSERT_THROW(connect(proxy, sink), EssentiaException);
}

TEST(Connectors, SinkProxy) {
  Source<string> src("Source1");
  SinkProxy<string> proxy("Proxy1");
  Sink<string> sink("Sink1");

  DBG("created");

  ASSERT_THROW(sink.available(), EssentiaException);
  ASSERT_THROW(proxy.available(), EssentiaException);
  DBG("attaching");
  attach(proxy, sink);
  DBG("attached");
  ASSERT_THROW(sink.available(), EssentiaException);
  ASSERT_THROW(proxy.available(), EssentiaException);

  DBG("connecting");
  connect(src, proxy);
  DBG("connecting ok");

  ASSERT_EQ(sink.available(), 0);
  ASSERT_EQ(proxy.available(), 0);

  DBG("pushing ok");
  src.push(string("hello"));
  src.push(string("world"));
  DBG("pushing ok");

  ASSERT_EQ(proxy.available(), 2);
  ASSERT_EQ(sink.available(), 2);

  EXPECT_EQ(sink.pop(), "hello");
  EXPECT_EQ(sink.pop(), "world");

  ASSERT_EQ(sink.available(), 0);
  ASSERT_THROW(sink.pop(), EssentiaException);

  // try with attach after connect, and disconnect, etc...
}

TEST(Connectors, Disconnect) {
  Source<string> src("Source1");
  SinkProxy<string> proxy("InputProxy");

  Sink<string> innerSink("InnerSink");
  Source<string> innerSource("InnerSource");

  SourceProxy<string> proxy2("OutputProxy");
  Sink<string> dest("Sink1");

  attach(proxy, innerSink);
  attach(innerSource, proxy2);

  connect(src, proxy);
  connect(proxy2, dest);

  ASSERT_EQ(innerSink.available(), 0);
  ASSERT_EQ(dest.available(), 0);

  src.push(string("foo"));
  src.push(string("bar"));

  innerSource.push(innerSink.pop());

  EXPECT_EQ(dest.pop(), "foo");
  ASSERT_EQ(dest.available(), 0);

  innerSource.push(innerSink.pop());

  EXPECT_EQ(dest.pop(), "bar");
  ASSERT_EQ(dest.available(), 0);

  DBG("flux ok");

  // reset all buffers
  src.reset(); innerSource.reset(); proxy2.reset();
  DBG("buffers reset");

  // re-attach new algo into proxies

  Sink<string> innerSink2("InnerSink2");
  Source<string> innerSource2("InnerSource2");

  detach(proxy, innerSink);

  ASSERT_EQ((size_t)1, src.sinks().size());

  attach(proxy, innerSink2);
  ASSERT_EQ((size_t)1, src.sinks().size());
  EXPECT_EQ(&proxy, src.sinks()[0]);

  detach(innerSource, proxy2);
  attach(innerSource2, proxy2);

  DBG("proxies attached");

  ASSERT_EQ(0, innerSink2.available());
  ASSERT_EQ(0, dest.available());

  DBG("pushing strings");

  src.push(string("foo"));
  src.push(string("bar"));

  DBG("transfer");

  innerSource2.push(innerSink2.pop());

  ASSERT_EQ(1, dest.available());
  EXPECT_EQ("foo", dest.pop());
  ASSERT_EQ(0, dest.available());

  innerSource2.push(innerSink2.pop());

  EXPECT_EQ("bar", dest.pop());
  ASSERT_EQ(0, dest.available());


}
// try:
//  - source || -> || source proxy -> [ sink -> source ] -> sink proxy || -> || sink
//  replace [ sink -> source ]
//  reset()
//  push again


TEST(Connectors, MultiConnectDisconnect) {
  Source<int> source1("source1"), source2("source2");
  Sink<int> sink1("sink1"), sink2("sink2"), sink3("sink3");

  SourceProxy<int> sourcep1("sourcep1"), sourcep2("sourcep2");
  SinkProxy<int> sinkp1("sinkp1"), sinkp2("sinkp2");

  DBG("k,jh");
  source1 >> sink1;
  source1 >> sink2;
  source1 >> sink3;

  ASSERT_EQ(3, source1.typedBuffer().numberReaders());
  ASSERT_EQ(2, sink3.id());

  disconnect(source1, sink1);
  disconnect(source1, sink2);

  ASSERT_EQ(1, source1.typedBuffer().numberReaders());
  ASSERT_EQ(0, sink3.id());

  source1.push(1);
  EXPECT_EQ(1, sink3.pop());
  disconnect(source1, sink3);
  source1.reset();

  DBG("yep0");
  source1 >> sinkp1;
  DBG("yep0.5");
  sourcep2 >> sink2;

  // should throw because no one is attached to the proxy
  DBG("yep");
  // pushing here works because our source is not a proxy
  //ASSERT_THROW(source1.push(3), EssentiaException);
  DBG("yep2");
  ASSERT_THROW(sink2.pop(), EssentiaException);
  DBG("yep3");

  sinkp1 >> sink1;
  source2 >> sourcep2;

  // should work now
  DBG("yep4");
  source1.push(23);
  DBG("yep5");
  ASSERT_EQ(23, sink1.pop());
  source2.push(27);
  DBG("yep6");
  ASSERT_EQ(27, sink2.pop());


  // connect the proxy to a source that has already a sink, so the proxy gets a reader ID of 1 instead of 0
  // re


  // try: - single src >> src proxy >> multi sinks
  //      - re-attach proxy to other source

  //

}

TEST(Connectors, ForwardSourceProxy) {
  Source<int> source1("source1");
  SourceProxy<int> sourcep1("sourcep1"), sourcep2("sourcep2");

  Sink<int> sink1("sink1"), sink2("sink2"), sink3("sink3"), sink4("sink4"), sink5("sink5"), sink6("sink6");

  /*

        /- Sink1
    Src1            /- Sink2
      | \- SrcProxy1            /- Sink3
      |             \- SrcProxy2
       \- Sink5               | \- Sink4
                              |
                               \- Sink6

  */

  sourcep2 >> sink3;
  sourcep2 >> sink4;

  ASSERT_THROW(sink3.pop(), EssentiaException);
  ASSERT_THROW(sink4.pop(), EssentiaException);

  sourcep1 >> sink2;
  sourcep1 >> sourcep2;

  ASSERT_THROW(sink2.pop(), EssentiaException);
  ASSERT_THROW(sink3.pop(), EssentiaException);
  ASSERT_THROW(sink4.pop(), EssentiaException);

  source1 >> sink1;
  source1 >> sourcep1;
  source1 >> sink5;
  sourcep2 >> sink6;

  EXPECT_EQ(0, sink1.id());
  EXPECT_EQ(1, sink2.id());
  EXPECT_EQ(2, sink3.id());
  EXPECT_EQ(3, sink4.id());
  EXPECT_EQ(4, sink5.id());
  EXPECT_EQ(5, sink6.id());

  EXPECT_EQ((size_t)3, sourcep2.sinks().size());
  EXPECT_EQ((size_t)4, sourcep1.sinks().size());
  EXPECT_EQ((size_t)6, source1.sinks().size());

  source1.push(23);

  EXPECT_EQ(23, sink1.pop());
  EXPECT_EQ(23, sink2.pop());
  EXPECT_EQ(23, sink3.pop());
  EXPECT_EQ(23, sink4.pop());
  EXPECT_EQ(23, sink5.pop());

  source1.push(24);

  EXPECT_EQ(24, sink1.pop());
  EXPECT_EQ(24, sink5.pop());

  disconnect(sourcep2, sink3);

  ASSERT_EQ((size_t)5, source1.sinks().size());
  ASSERT_EQ((size_t)3, sourcep1.sinks().size());
  ASSERT_EQ((size_t)2, sourcep2.sinks().size());

  EXPECT_EQ(2, sink4.id());
  EXPECT_EQ(3, sink5.id());

  ASSERT_THROW(sink5.pop(), EssentiaException);
  EXPECT_EQ(24, sink2.pop());
  EXPECT_EQ(24, sink4.pop());

  sink3.reset();
  sourcep2 >> sink3;

  source1.push(25);
  EXPECT_EQ(25, sink3.pop());
  EXPECT_EQ(23, sink6.pop());
  EXPECT_EQ(24, sink6.pop());
  EXPECT_EQ(25, sink6.pop());

}

TEST(Connectors, ForwardSinkProxy) {
  Source<int> source1("source1");
  SinkProxy<int> sinkp1("sinkp1"), sinkp2("sinkp2");

  Sink<int> sink1("sink1"), sink2("sink2"), sink3("sink3");

  /*

        /- Sink1
    Src1
      | \- SinkProxy1 - SinkProxy2 - Sink2
      |
       \- Sink3

  */


  sinkp2 >> sink2;

  ASSERT_THROW(sink2.pop(), EssentiaException);

  source1 >> sink1;
  source1 >> sinkp1;
  source1 >> sink3;

  source1.push(7);
  EXPECT_EQ(7, sink1.pop());
  //ASSERT_THROW(sinkp1.pop(), EssentiaException); // pop not defined for sinkproxy
  EXPECT_EQ(7, sink3.pop());

  source1.push(8);

  sinkp1 >> sinkp2;  // token 7 and 8 are not lost by now, as sinkp1 already had a valid reader ID when they were produced

  source1.push(9);

  EXPECT_EQ(8, sink1.pop());
  EXPECT_EQ(9, sink1.pop());
  EXPECT_EQ(7, sink2.pop());
  EXPECT_EQ(8, sink2.pop());
  EXPECT_EQ(9, sink2.pop());
  EXPECT_EQ(8, sink3.pop());
  EXPECT_EQ(9, sink3.pop());
}

TEST(Connectors, KindaComplexMess) {
  Source<int>* source1 = new Source<int>("source1");

  SourceProxy<int>* srcproxy1 = new SourceProxy<int>("srcproxy1");
  SourceProxy<int>* srcproxy2 = new SourceProxy<int>("srcproxy2");

  SinkProxy<int>* sinkproxy1 = new SinkProxy<int>("sinkproxy1");
  SinkProxy<int>* sinkproxy2 = new SinkProxy<int>("sinkproxy2");
  SinkProxy<int>* sinkproxy3 = new SinkProxy<int>("sinkproxy3");

  Sink<int>* sink1 = new Sink<int>("sink1");
  Sink<int>* sink2 = new Sink<int>("sink2");
  Sink<int>* sink3 = new Sink<int>("sink3");
  Sink<int>* sink4 = new Sink<int>("sink4");
  Sink<int>* sink5 = new Sink<int>("sink5");
  Sink<int>* sink6 = new Sink<int>("sink6");
  Sink<int>* sink7 = new Sink<int>("sink7");

  /*

        /- Sink1
    Src1            /- Sink2
      | \- SrcProxy1            /- Sink3
      |           | \- SrcProxy2
       \- Sink5   |           | \- SinkProxy1 - SinkProxy2 - Sink4
                  |           |
                  |            \- Sink6
                   \
                    \- SinkProxy3 - Sink7
  */

  *source1 >> *sink1;
  *source1 >> *srcproxy1;
  *srcproxy1 >> *sink2;
  ASSERT_EQ(1, sink2->id());

  *srcproxy2 >> *sink3;
  //ASSERT_THROW(sink3->id(), EssentiaException);

  *srcproxy1 >> *srcproxy2;
  ASSERT_EQ(2, sink3->id());

  *sinkproxy2 >> *sink4;
  ASSERT_THROW(sink4->id(), EssentiaException);

  *srcproxy2 >> *sinkproxy1;
  *sinkproxy1 >> *sinkproxy2;
  ASSERT_EQ(3, sink4->id());
  ASSERT_EQ(sink4->source(), srcproxy2);

  *source1 >> *sink5;
  *srcproxy2 >> *sink6;
  ASSERT_EQ(5, sink6->id());

  *srcproxy1 >> *sinkproxy3;
  ASSERT_EQ(6, sinkproxy3->id());

  *sinkproxy3 >> *sink7;

  ASSERT_EQ(0, sink1->id());
  ASSERT_EQ(1, sink2->id());
  ASSERT_EQ(2, sink3->id());
  ASSERT_EQ(3, sink4->id());
  ASSERT_EQ(4, sink5->id());
  ASSERT_EQ(5, sink6->id());
  ASSERT_EQ(6, sink7->id());

  source1->push(1);
  source1->push(2);

  EXPECT_EQ(1, sink2->pop());
  EXPECT_EQ(1, sink4->pop());
  EXPECT_EQ(1, sink5->pop());

  // delete SinkProxy1:
  //  - sinkproxy2 and sink4 should be disconnected
  //  - sinkproxy3 as well as sink5, sink6 and sink7 get their IDs reassigned
  delete sinkproxy1;

  /*

        /- Sink1
    Src1            /- Sink2
      | \- SrcProxy1            /- Sink3
      |           | \- SrcProxy2
       \- Sink5   |           |                 SinkProxy2 - Sink4
                  |           |
                  |            \- Sink6
                   \
                    \- SinkProxy3 - Sink7
  */

  ASSERT_THROW(sinkproxy2->id(), EssentiaException);
  ASSERT_THROW(sink4->pop(), EssentiaException);
  ASSERT_THROW(sink4->id(), EssentiaException);

  ASSERT_EQ(3, sink5->id());
  ASSERT_EQ(sink5, source1->sinks()[3]);
  ASSERT_EQ(4, sink6->id());
  ASSERT_EQ(sink6, source1->sinks()[4]);
  ASSERT_EQ(5, sink7->id());
  ASSERT_EQ(5, sinkproxy3->id());
  ASSERT_EQ(sinkproxy3, source1->sinks()[5]);

  EXPECT_EQ(2, sink2->pop());
  EXPECT_EQ(1, sink3->pop());
  EXPECT_EQ(1, sink7->pop());
  EXPECT_EQ(2, sink5->pop());
  EXPECT_EQ(1, sink6->pop());

  detach(*source1, *srcproxy1);
  ASSERT_EQ((size_t)2, source1->sinks().size());
  ASSERT_EQ(sink1, source1->sinks()[0]);
  ASSERT_EQ(sink5, source1->sinks()[1]);

  ASSERT_EQ((size_t)4, srcproxy1->sinks().size());
  ASSERT_EQ(sink2, srcproxy1->sinks()[0]);
  ASSERT_EQ(sink3, srcproxy1->sinks()[1]);
  ASSERT_EQ(sink6, srcproxy1->sinks()[2]);
  ASSERT_EQ(sinkproxy3, srcproxy1->sinks()[3]);
  // the following need not be correct as we're not connected to a buffer anymore anyway
  //ASSERT_EQ(3, sinkproxy3->id());
  //ASSERT_EQ(3, sink7->id());
  ASSERT_THROW(sink7->pop(), EssentiaException);

  *source1 >> *srcproxy1;
  ASSERT_EQ((size_t)6, source1->sinks().size());
  ASSERT_EQ(sink1, source1->sinks()[0]);
  ASSERT_EQ(sink5, source1->sinks()[1]);
  ASSERT_EQ(sink2, source1->sinks()[2]);
  ASSERT_EQ(sink3, source1->sinks()[3]);
  ASSERT_EQ(sink6, source1->sinks()[4]);
  ASSERT_EQ(sinkproxy3, source1->sinks()[5]);

  ASSERT_EQ(0, sink1->id());
  ASSERT_EQ(1, sink5->id());
  ASSERT_EQ(2, sink2->id());
  ASSERT_EQ(3, sink3->id());
  ASSERT_EQ(4, sink6->id());
  ASSERT_EQ(5, sink7->id());

  EXPECT_EQ(1, sink1->pop());
  ASSERT_THROW(sink5->pop(), EssentiaException); // no more available

  source1->push(3);
  EXPECT_EQ(2, sink1->pop());
  EXPECT_EQ(3, sink1->pop());
  EXPECT_EQ(3, sink2->pop());
  EXPECT_EQ(3, sink3->pop());
  EXPECT_EQ(3, sink5->pop());
  EXPECT_EQ(3, sink6->pop());
  EXPECT_EQ(3, sink7->pop());

  delete sinkproxy2;
  delete sink4;

  delete sink7;
  delete sinkproxy3;
  delete sink2;
  delete srcproxy1;
  delete srcproxy2;
  delete sink3;
  delete sink6;
  delete sink1;
  delete source1;
  delete sink5;
}
