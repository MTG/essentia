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

#include "SuperFluxNovelty.h"
#include "essentiamath.h"



namespace essentia {
    namespace standard {
        
        
        const char* SuperFluxNovelty::name = "SuperFluxNovelty";
        const char* SuperFluxNovelty::description = DOC("Novelty curve from Superflux algorithm (see SuperFluxExtractor for references)");
        
        
        void SuperFluxNovelty::configure() {
            
            
            
            _binW = parameter("binWidth").toInt();
            _maxf->configure("width",_binW,"causal",false);
            _frameWi = parameter("frameWidth").toInt();
            
            
            
        }
        
        
        void SuperFluxNovelty::compute() {
            
            const vector< vector<Real> >& bands = _bands.get();
            
            Real& diffs = _diffs.get();
            
            int nFrames = bands.size();
            if(!nFrames){
                throw EssentiaException("SuperFluxNovelty : empty frames");
            }
            int nBands= bands[0].size();
            if(!nBands){
                throw EssentiaException("SuperFluxNovelty : empty bands ");
            }
            
            if(_frameWi>=nFrames){
                
                throw EssentiaException("SuperFluxNovelty : no enough frames comparing to frame witdh");
            }
            
            vector<Real> maxsBuffer(nBands,0);
            
            // buffer for differences
            Real cur_diff;
            diffs=0;
            for (int i = _frameWi ; i< nFrames;i++){
                
                
                
                _maxf->input("signal").set(bands[i-_frameWi]);
                _maxf->output("signal").set(maxsBuffer);
                _maxf->compute();
                
                cur_diff = 0;
                
                for (int j = 0;j<nBands;j++){
                    cur_diff= bands[i][j]-maxsBuffer[j];
                    if(cur_diff>0.0){
                        diffs +=cur_diff ;
                    }
                }
                
                
            }
            
            
            
            return;
        }
        
        void SuperFluxNovelty::reset() {
            Algorithm::reset();
            
        }
        
        
        
        
    } // namespace standard
} // namespace essentia




namespace essentia {
    namespace streaming {
        
        const char* SuperFluxNovelty::name = standard::SuperFluxNovelty::name;
        const char* SuperFluxNovelty::description = standard::SuperFluxNovelty::description;
        
        
        AlgorithmStatus SuperFluxNovelty::process() {
            
            
            
            AlgorithmStatus status = acquireData();
            if (status != OK) {
                
                return status;
            }
            
            
            _algo->input("bands").set(_bands.tokens());
            _algo->output("Differences").set(_diffs.firstToken());
            
            _algo->compute();
            
            // give back the tokens that were reserved
            releaseData();
            
            return OK;
            
        }
        
        
    } // namespace streaming
} // namespace essentia
