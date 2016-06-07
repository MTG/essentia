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

#include <iostream>
#include <essentia/algorithmfactory.h>
#include <essentia/essentiamath.h> // for the isSilent function
#include <essentia/pool.h>

using namespace std;
using namespace essentia;
using namespace standard;


int main(int argc, char* argv[]) {

    if (argc < 3) {
        cout << "Error: wrong number of arguments" << endl;
        cout << "Usage: " << argv[0] << " input_audiofile output_csvfile [0...5 algorithm]" << endl;
        cout << "0: monophonic pitch estimation Yin FFT" << endl;
        cout << "1: monophonic pitch estimation Melodia" << endl;
        cout << "2: monophonic note transcription" << endl;
        cout << "3: multi pitch estimation Klapuri" << endl;
        cout << "4: multi pitch estimation Melodia" << endl;
        cout << "5: predominant pitch estimation Melodia" << endl;
        exit(1);
    }
    

    int algo=atoi(argv[3]);
    
    /////// PARAMS //////////////
    int framesize = 2048;
    int hopsize = 128;
    int sr = 44100;
    
    // register the algorithms in the factory(ies)
    essentia::init();
    
    // instanciate factory
    AlgorithmFactory& factory = AlgorithmFactory::instance();
    
    // audio loader (we always need it...)
    Algorithm* audioload = factory.create("MonoLoader",
                                          "filename", argv[1],
                                          "sampleRate", sr,
                                          "downmix", "mix");

    vector<Real> audio;
    audioload->output("audio").set(audio);
    audioload->compute();
    
    switch (algo) {
            
        /* ------ monophonic pitch Yin FFT ----------- */
        case 0:
        {
            cout << "monophonic pitch estimation Yin FFT" << endl;
            
            
            // create algorithms
            Algorithm* frameCutter = factory.create("FrameCutter",
                                                    "frameSize", framesize,
                                                    "hopSize", hopsize,
                                                    "startFromZero", false);
            
            Algorithm* window = factory.create("Windowing",
                                               "type", "hann",
                                               "zeroPadding", 0);
            
            Algorithm* spectrum = factory.create("Spectrum",
                                                 "size", framesize);
            
            Algorithm* pitchDetect = factory.create("PitchYinFFT",
                                                    "frameSize", framesize,
                                                    "sampleRate", sr);
            
            
            // configure frameCutter:
            vector<Real> frame;
            frameCutter->input("signal").set(audio);
            frameCutter->output("frame").set(frame);
            
            // configure windowing:
            vector<Real> windowedframe;
            window->input("frame").set(frame);
            window->output("frame").set(windowedframe);
            
            // configure spectrum:
            vector<Real> spec;
            spectrum->input("frame").set(windowedframe);
            spectrum->output("spectrum").set(spec);
            
            // configure pitch extraction:
            Real thisPitch = 0., thisConf = 0;
            Real localTime=0.;
            vector<Real> allPitches, allConf, time;
            pitchDetect->input("spectrum").set(spec);
            pitchDetect->output("pitch").set(thisPitch);
            pitchDetect->output("pitchConfidence").set(thisConf);
            
            
            // process:
            while (true) {
                frameCutter->compute();
                
                if (!frame.size())
                    break;
                
                if (isSilent(frame))
                    continue;
                
                window->compute();
                spectrum->compute();
                pitchDetect->compute();
                allPitches.push_back(thisPitch);
                localTime+=float(hopsize)/float(sr);
                time.push_back(localTime);
                allConf.push_back(thisConf);
            }
            
            // write to csv file
            ofstream outfile(argv[2]);
            cout << "time   pitch [Hz]  pitch confidence" << endl;
            for (int i=0; i<(int)time.size(); i++){
                outfile << time[i] << ", " << allPitches[i] << ", " << allConf[i] << endl;
            }
            outfile.close();
            
            // clean up
            delete frameCutter;
            delete spectrum;
            delete pitchDetect;
            
            break;
        }
        /* -------------------------------------------- */
            
            
            
        case 1:
        {
            cout << "1: monophonic pitch estimation Melodia" << endl;
            
            // create algorithms
            Algorithm* pitchDetect = factory.create("PitchMelodia",
                                                    "sampleRate", sr, "hopSize", hopsize, "frameSize", framesize);
            
            // configure
            vector<Real> pitch, pitchConfidence;
            pitchDetect->input("signal").set(audio);
            pitchDetect->output("pitch").set(pitch);
            pitchDetect->output("pitchConfidence").set(pitchConfidence);
            
            // process
            pitchDetect->compute();
            
            // write to csv file
            ofstream outfile(argv[2]);
            cout << "time   pitch [Hz]  pitch confidence" << endl;
            for (int i=0; i<(int)pitch.size(); i++){
                outfile << float(i)*float(hopsize)/float(sr) << ", " << pitch[i] << ", " << pitchConfidence[i] << endl;
            }
            outfile.close();
            
            // clean up
            delete pitchDetect;
            
            break;
        }
            
        case 2:
        {
            cout << "2: monophonic note transcription" << endl;
            
            // create algorithms
            Algorithm* pitchDetect = factory.create("PitchMelodia",
                                                    "sampleRate", sr, "hopSize", hopsize, "frameSize", framesize);
            
            Algorithm* noteSeg = factory.create("PitchContourSegmentation", "sampleRate", sr, "hopSize", hopsize);
            
            // configure pitch detection
            vector<Real> pitch, pitchConfidence;
            pitchDetect->input("signal").set(audio);
            pitchDetect->output("pitch").set(pitch);
            pitchDetect->output("pitchConfidence").set(pitchConfidence);
            
            // configure note segmentation
            vector<Real> onset, duration, MIDIpitch;
            noteSeg->input("pitch").set(pitch);
            noteSeg->input("signal").set(audio);
            noteSeg->output("onset").set(onset);
            noteSeg->output("duration").set(duration);
            noteSeg->output("MIDIpitch").set(MIDIpitch);
            
            // process:
            // extract monophonic melody using the MELODIA algorithm
            pitchDetect->compute();
            
            // segment the resulting contour into discrete note events
            noteSeg->compute();
            
            // write to csv file
            ofstream outfile(argv[2]);
            cout << "onset [s]   duration [s]  MIDI pitch" << endl;
            for (int i=0; i<(int)MIDIpitch.size(); i++){
                outfile << onset[i] << ", " << duration[i] << ", " << MIDIpitch[i] << endl;
            }
            outfile.close();
            
            // clean up
            delete pitchDetect;
            delete noteSeg;
            
            break;
        }
            
        case 3:
        {
            cout << "3: multi pitch estimation Klapuri" << endl;
            
            // create algorithm
            Algorithm* multiPitch = factory.create("MultiPitchKlapuri", "sampleRate", sr);
            
            // configure algorithm
            vector<vector<Real> >pitchMulti;
            multiPitch->input("signal").set(audio);
            multiPitch->output("pitch").set(pitchMulti);
            
            // process:
            multiPitch->compute();
            
            // write to file
            ofstream outfile(argv[2]);
            cout << "time   pitch [Hz]" << endl;
            for (int ii=0; ii<(int)pitchMulti.size(); ii++){
                for (int jj=0; jj<(int)pitchMulti[ii].size(); jj++){
                    outfile <<  float(ii)*hopsize/sr << ", " << pitchMulti[ii][jj] << endl;
                }
            }
            outfile.close();
            
            // clean up
            delete multiPitch;
            
            break;
        }
        
        case 4:
        {
            cout << "4: multi pitch estimation Melodia" << endl;
            
            // create algorithms
            Algorithm* predmelMulti = factory.create("MultiPitchMelodia", "sampleRate", sr, "frameSize", framesize, "hopSize", hopsize);
            Algorithm* el = factory.create("EqualLoudness","sampleRate", sr);
            
            // configure
            vector<Real> audioEQ;
            vector<vector<Real> >pitchMulti;
            el->input("signal").set(audio);
            el->output("signal").set(audioEQ);
            predmelMulti->input("signal").set(audioEQ);
            predmelMulti->output("pitch").set(pitchMulti);
            
            // process:
            el->compute();
            predmelMulti->compute();
            
            // write to file
            ofstream outfile(argv[2]);
            cout << "time   pitch [Hz]" << endl;
            for (int ii=0; ii<(int)pitchMulti.size(); ii++){
                for (int jj=0; jj<(int)pitchMulti[ii].size(); jj++){
                    outfile <<  float(ii)*float(hopsize)/float(sr) << ", " << pitchMulti[ii][jj] << endl;
                }
            }
            outfile.close();
            
            break;
        }
            
        case 5:
        {
            cout << "5: predominant pitch estimation Melodia" << endl;
            
            // create algorithms
            Algorithm* pitchDetect = factory.create("PredominantPitchMelodia",
                                                    "sampleRate", sr, "hopSize", hopsize, "frameSize", framesize);
            
            
            // configure pitch detection
            vector<Real> pitch, pitchConfidence;
            pitchDetect->input("signal").set(audio);
            pitchDetect->output("pitch").set(pitch);
            pitchDetect->output("pitchConfidence").set(pitchConfidence);

            
            // process:
            // extract predominant melody using the MELODIA algorithm
            pitchDetect->compute();
            
            // write to csv file
            ofstream outfile(argv[2]);
            cout << "time [s]   pitch [Hz]  pitch confidence" << endl;
            for (int i=0; i<(int)pitch.size(); i++){
                outfile <<  float(i)*hopsize/sr << ", " << pitch[i] << ", " << pitchConfidence[i] << endl;
            }
            outfile.close();
            
            // clean up
            delete pitchDetect;

            break;
        }
            
        default:
        {
            cout << "Usage: " << argv[0] << " input_audiofile output_csvfile [0...5 algorithm]" << endl;
            cout << "0: monophonic pitch estimation Yin FFT" << endl;
            cout << "1: monophonic pitch estimation Melodia" << endl;
            cout << "2: multi pitch estimation Klapuri" << endl;
            cout << "3: multi pitch estimation Melodia" << endl;
            cout << "4: predominant pitch estimation Melodia" << endl;
            cout << "5: monophonic note transcription" << endl;
            break;
        }
    }

  
    
  
    // clean up
    delete audioload;
    
    // shut down essentia
    essentia::shutdown();
    
    return 0;
}
