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
#ifndef ESSENTIA_COVERSONGSIMILARITY_H
#define ESSENTIA_COVERSONGSIMILARITY_H
#include "algorithmfactory.h"
#include <complex>
namespace essentia {
namespace standard {
 class CoverSongSimilarity : public Algorithm {
  protected:
  Input<std::vector<std::vector<Real>> > _inputArray;
  //Output<Real> _similarityMeasure;
  Output<std::vector<std::vector<Real>> > _scoreMatrix;
  double gammaO;
  double gammaE;
  public:
  CoverSongSimilarity() {
    declareInput(_inputArray, "inputArray", " a cross recurrent plot (2d binary similarity matrix) of two audio chroma vector as the input array");
    //declareOutput(_similarityMeasure, "similarityMeasure", "Qmax cover song similarity measure (distance) from the input cross recurrent plot");
    declareOutput(_scoreMatrix, "scoreMatrix", "2d Qmax cover song similarity scoring matrix from the input cross recurrent plot");
   }

   void declareParameters() {
    declareParameter("gammaO", "penalty for disurption onset", "[0,inf)", 0.5);
    declareParameter("gammaE", "penalty for disurption extension", "[0,inf)", 0.5);
    declareParameter("simType", "type of cover song similarity measure", "{qmax, dmax}", "qmax");
  }

  void configure();
  void compute();
  static const char* name;
  static const char* category;
  static const char* description;

  float _gammaO;
  float _gammaE;
  enum SimType {
    QMAX, DMAX
  };
  SimType _simType;

 };
 } // namespace standard
} // namespace essentia
 #include "streamingalgorithmwrapper.h"
 namespace essentia {
namespace streaming {
 class CoverSongSimilarity : public StreamingAlgorithmWrapper {
  protected:
  Sink<std::vector<Real> > _inputArray;
  //Source<Real> _similarityMeasure;
  Source<std::vector<std::vector<Real>> > _scoreMatrix;
  public:
  CoverSongSimilarity() {
    declareAlgorithm("CoverSongSimilarity");
    declareInput(_inputArray, TOKEN, "inputArray");
    //declareOutput(_similarityMeasure, TOKEN, "similarityMeasure");
    declareOutput(_scoreMatrix, TOKEN, "scoreMatrix");
  }
};
 } // namespace streaming
} // namespace essentia
 #endif // ESSENTIA_HISTOGRAM_H
