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

#include "algorithmfactory.h"
#include "../algorithms/io/ringbufferinput.h"
#include "../algorithms/io/ringbufferoutput.h"
#include "diskwriter.h"

#ifdef WIN32
	#include <windows.h>
	#include <pthread.h>
	#define usleep(x) Sleep((x)/1000)
#else
	#include <unistd.h>
#endif


using namespace std;
using namespace essentia;
using namespace essentia::streaming;


//      int  pthread_create(pthread_t  *  thread, pthread_attr_t * attr, void *
//       (*start_routine)(void *), void * arg);

void* fill_thread(void* ptr)
{
	Real test[1024];
	int j = 0;
	RingBufferInput* ringBufferInput = static_cast<RingBufferInput*>(ptr);
	while (1)
	{
		for (int i=0;i<441;i++) test[i] = j++;

		ringBufferInput->add(test,441);
		usleep(10000);
	}
	return 0;
}

void* empty_thread(void* ptr)
{
	Real buf[1024];
	RingBufferOutput* ringBufferOutput = static_cast<RingBufferOutput*>(ptr);
	while (1)
	{
		int n = ringBufferOutput->get(buf,1024);
		cout << n << " values:";
		for (int i=0;i<n;i++)
		{
			cout << " " << buf[i];
		}
		cout << endl;
		usleep(10000);
	}
	return 0;
}


int main(int argc,char** argv)
{
	if (argc>2 || (argc==2 && string(argv[1])=="-h"))
	{
		cout << "Error: wrong number of arguments" << endl;
		cout << "Usage: " << argv[0] << " [output_filename]" << endl;
		cout <<
"  A thread fills a RingBufferInput with test data. This RingBufferInput form"
<< endl <<
"  part of a streaming network that will write the data to disk."
<< endl <<
"  If no output_filename is specified, a RingBufferOutput algorithm will be"
<< endl <<
"  connected, which will be emptied by a thread that writes the data to the"
<< endl <<
"standard output." << endl;
		exit(1);
	}
  string outputFilename;
	if (argc==2)
	{
		outputFilename = argv[1];
	}

	essentia::init();

  StreamingAlgorithm* ringBufferInput = streaming::AlgorithmFactory::create("RingBufferInput");
  StreamingAlgorithm* ringBufferOutput = streaming::AlgorithmFactory::create("RingBufferOutput");
	StreamingAlgorithm* frameCutter = 0;
	DiskWriter< vector<Real> >* diskWriter = 0;
	if (outputFilename!="")
	{
		frameCutter = streaming::AlgorithmFactory::create("FrameCutter","startFromZero",true,"frameSize",128,"hopSize",128);
		diskWriter = new DiskWriter< vector<Real> >(outputFilename);
		connect(ringBufferInput->output("signal"), frameCutter->input("signal"));
		connect(frameCutter->output("frame"), diskWriter->input("data"));
	}
	else
	{
		connect(ringBufferInput->output("signal"), ringBufferOutput->input("signal"));
	}

	pthread_t fill_thr;
	pthread_create(&fill_thr, 0, fill_thread, ringBufferInput);

	if (ringBufferOutput)
	{
		pthread_t empty_thr;
		pthread_create(&empty_thr, 0, empty_thread, ringBufferOutput);
	}

	runGenerator(ringBufferInput);

}	
