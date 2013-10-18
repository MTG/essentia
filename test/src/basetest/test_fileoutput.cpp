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

#include <complex>
#include "essentia_gtest.h"
#include "network.h"
#include "vectorinput.h"
#include "fileoutput.h"
using namespace std;
using namespace essentia;
using namespace essentia::streaming;


TEST(FileOutput, Real) {
  string filename = "build/test/fileoutput.txt";
  Real inputData[] = {0.0, 0.25, 0.5, 0.75, 1.0};

  Algorithm* gen = new VectorInput<Real>(inputData);
  Algorithm* file = new FileOutput<Real>();
  file->configure("filename", filename, "mode", "text");

  connect(gen->output("data"), file->input("data"));

  scheduler::Network(gen).run();

  ifstream fileStream;
  fileStream.open(filename.c_str());
  string line;
  vector<Real> vectorLine = arrayToVector<Real>(inputData);
  vector<Real> results;

  // read results
  while (getline(fileStream, line)) {
    results.push_back(atof(line.c_str()));
  }

  // clean up
  fileStream.close();
  if (remove(filename.c_str())) throw EssentiaException("TestFileOutput: Error deleting ", filename);

  // verify results
  EXPECT_VEC_EQ(vectorLine, results);
}


TEST(FileOutput, String) {
  string filename = "build/test/fileoutput.txt";
  const char* inputData[] = {"foo", "bar"};

  Algorithm* gen = new VectorInput<string>(inputData);
  Algorithm* file = new FileOutput<string>();
  file->configure("filename", filename, "mode", "text");

  connect(gen->output("data"), file->input("data"));

  scheduler::Network(gen).run();

  ifstream fileStream;
  fileStream.open(filename.c_str());
  string line;
  vector<string> vectorLine = arrayToVector<string>(inputData);
  vector<string> results;

  // read results
  while (getline(fileStream, line)) {
    results.push_back(line);
  }

  // clean up
  fileStream.close();
  if (remove(filename.c_str())) throw EssentiaException("TestFileOutput: Error deleting ", filename);

  // verify results
  EXPECT_VEC_EQ(vectorLine, results);
}


TEST(FileOutput, VectorReal) {
  string filename = "build/test/fileoutput.txt";
  Real inputData0[] = {0.0, 0.25, 0.5, 0.75};
  Real inputData1[] = {1.0, 1.25, 1.5, 1.75};
  Real inputData2[] = {2.0, 2.25, 2.5, 2.75};
  vector<vector<Real> > inputData;
  inputData.push_back(arrayToVector<Real>(inputData0));
  inputData.push_back(arrayToVector<Real>(inputData1));
  inputData.push_back(arrayToVector<Real>(inputData2));

  streaming::Algorithm* gen = new VectorInput<vector<Real> >(&inputData);
  streaming::Algorithm* file = new FileOutput<vector<Real> >();
  file->configure("filename", filename, "mode", "text");

  connect(gen->output("data"), file->input("data"));

  scheduler::Network(gen).run();

  ifstream fileStream;
  fileStream.open(filename.c_str());
  vector<string> lines;
  string line;
  while (getline(fileStream, line)) lines.push_back(line);

  fileStream.close();
  if (remove(filename.c_str())) throw EssentiaException("TestFileOutput: Error deleting ", filename);

  const char* expected[] = { "[0, 0.25, 0.5, 0.75]",
                             "[1, 1.25, 1.5, 1.75]",
                             "[2, 2.25, 2.5, 2.75]" };

  EXPECT_VEC_EQ(arrayToVector<string>(expected), lines);
}


TEST(FileOutput, VectorString) {
  string filename = "build/test/fileoutput.txt";
  const char* inputData0[] = {"foo0", "foo01"};
  const char* inputData1[] = {"foo1", "foo11"};
  const char* inputData2[] = {"foo2", "foo21"};
  vector<vector<string> > inputData;
  inputData.push_back(arrayToVector<string>(inputData0));
  inputData.push_back(arrayToVector<string>(inputData1));
  inputData.push_back(arrayToVector<string>(inputData2));

  streaming::Algorithm* gen = new VectorInput<vector<string> >(&inputData);
  streaming::Algorithm* file = new FileOutput<vector<string> >();
  file->configure("filename", filename, "mode", "text");

  connect(gen->output("data"), file->input("data"));

  scheduler::Network(gen).run();

  ifstream fileStream;
  fileStream.open(filename.c_str());
  vector<string> lines;
  string line;
  while (getline(fileStream, line)) lines.push_back(line);

  fileStream.close();
  if (remove(filename.c_str())) throw EssentiaException("TestFileOutput: Error deleting ", filename);

  const char* expected[] = { "[foo0, foo01]", "[foo1, foo11]", "[foo2, foo21]" };

  EXPECT_VEC_EQ(arrayToVector<string>(expected), lines);
}


TEST(FileOutput, Complex) {
  string filename = "build/test/fileoutput.txt";
  complex<Real> inputData[] = {complex<Real>(0,0), complex<Real>(0,1), complex<Real>(0,2), complex<Real>(0,3)};

  streaming::Algorithm* gen = new VectorInput<complex<Real> >(inputData);
  streaming::Algorithm* file = new FileOutput<complex<Real> >();
  file->configure("filename", filename, "mode", "text");

  connect(gen->output("data"), file->input("data"));

  scheduler::Network(gen).run();

  ifstream fileStream;
  fileStream.open(filename.c_str());
  vector<string> lines;
  string line;
  while (getline(fileStream, line)) lines.push_back(line);

  fileStream.close();
  if (remove(filename.c_str())) throw EssentiaException("TestFileOutput: Error deleting ", filename);

  const char* expected[] = { "(0,0)", "(0,1)", "(0,2)", "(0,3)" };
  EXPECT_VEC_EQ(arrayToVector<string>(expected), lines);
}


TEST(FileOutput, VectorComplexReal) {
  string filename = "build/test/fileoutput.txt";
  complex<Real> inputData0[] = {complex<Real>(0,0), complex<Real>(0,1), complex<Real>(0,2), complex<Real>(0,3)};
  complex<Real> inputData1[] = {complex<Real>(1,0), complex<Real>(1,1), complex<Real>(1,2), complex<Real>(1,3)};
  complex<Real> inputData2[] = {complex<Real>(2,0), complex<Real>(2,1), complex<Real>(2,2), complex<Real>(2,3)};
  vector<vector<complex<Real> > > inputData;
  inputData.push_back(arrayToVector<complex<Real> >(inputData0));
  inputData.push_back(arrayToVector<complex<Real> >(inputData1));
  inputData.push_back(arrayToVector<complex<Real> >(inputData2));

  streaming::Algorithm* gen = new VectorInput<vector<complex<Real> > >(&inputData);
  streaming::Algorithm* file = new FileOutput<vector<complex<Real> > >();
  file->configure("filename", filename, "mode", "text");

  connect(gen->output("data"), file->input("data"));

  scheduler::Network(gen).run();

  ifstream fileStream;
  fileStream.open(filename.c_str());
  vector<string> lines;
  string line;
  while (getline(fileStream, line)) lines.push_back(line);

  fileStream.close();
  if (remove(filename.c_str())) throw EssentiaException("TestFileOutput: Error deleting ", filename);

  const char* expected[] = { "[(0,0), (0,1), (0,2), (0,3)]",
                             "[(1,0), (1,1), (1,2), (1,3)]",
                             "[(2,0), (2,1), (2,2), (2,3)]" };

  EXPECT_VEC_EQ(arrayToVector<string>(expected), lines);
}


TEST(FileOutput, Array1D) {
  string filename = "build/test/fileoutput.txt";
  int size = 2;
  TNT::Array1D<Real> inputData0(size);
  TNT::Array1D<Real> inputData1(size);
  TNT::Array1D<Real> inputData2(size);

  // initialise arrays:
  int i=0;
  for (; i<size; i++) inputData0[i]=float(i)/float(size);
  for (; i<size*2; i++) inputData1[i-size]=float(i)/float(size);
  for (; i<size*3; i++) inputData2[i-2*size]=float(i)/float(size);

  vector<TNT::Array1D<Real> > inputData;
  inputData.push_back(inputData0);
  inputData.push_back(inputData1);
  inputData.push_back(inputData2);

  streaming::Algorithm* gen = new VectorInput<TNT::Array1D<Real> >(&inputData);
  streaming::Algorithm* file = new FileOutput<TNT::Array1D<Real> >();
  file->configure("filename", filename, "mode", "text");

  connect(gen->output("data"), file->input("data"));

  scheduler::Network(gen).run();

  ifstream fileStream;
  fileStream.open(filename.c_str());
  vector<string> lines;
  string line;
  while (getline(fileStream, line)) lines.push_back(line);

  fileStream.close();
  if (remove(filename.c_str())) throw EssentiaException("TestFileOutput: Error deleting ", filename);

  // streaming array1d adds 3 more lines: 1 for dimension + 2 returns:
  EXPECT_EQ(int(lines.size()), (size+3)*3);
  i = 0;
  // inputData0
  EXPECT_EQ(atoi(lines[i++].c_str()), inputData0.dim()); // dimension
  for (int j=0; j<size; j++) EXPECT_EQ(atof(lines[i++].c_str()), inputData0[j]);
  for (int j=0; j<2; j++) EXPECT_EQ(lines[i++], ""); // two returns

  // inputData1
  EXPECT_EQ(atoi(lines[i++].c_str()), inputData1.dim()); // dimension
  for (int j=0; j<size; j++) EXPECT_EQ(atof(lines[i++].c_str()), inputData1[j]);
  for (int j=0; j<2; j++) EXPECT_EQ(lines[i++], ""); // two returns

  // inputData2
  EXPECT_EQ(atoi(lines[i++].c_str()), inputData1.dim()); // dimension
  for (int j=0; j<size; j++) EXPECT_EQ(atof(lines[i++].c_str()), inputData2[j]);
  for (int j=0; j<2; j++) EXPECT_EQ(lines[i++], ""); // two returns
}


TEST(FileOutput, Array2D) {
  string filename = "build/test/fileoutput.txt";
  int size = 3;     // number of matrices
  int M = 2, N = 3; // matrix dimensions
  vector<TNT::Array2D<Real> > inputData;
  inputData.reserve(size);
  // init vector:
  int count=0;
  for (int i=0; i<size; i++) {
    TNT::Array2D<Real> mat(M,N);
    for (int row=0; row<M; row++) {
      for (int col=0; col<N; col++, count++) {
        mat[row][col]=float(count)/float(M*N);
      }
    }
    inputData.push_back(mat);
  }

  streaming::Algorithm* gen = new VectorInput<TNT::Array2D<Real> >(&inputData);
  streaming::Algorithm* file = new FileOutput<TNT::Array2D<Real> >();
  file->configure("filename", filename, "mode", "text");

  connect(gen->output("data"), file->input("data"));

  scheduler::Network(gen).run();

  ifstream fileStream;
  fileStream.open(filename.c_str());
  vector<string> lines;
  string line;
  while (getline(fileStream, line)) lines.push_back(line);

  fileStream.close();
  if (remove(filename.c_str())) throw EssentiaException("TestFileOutput: Error deleting ", filename);

  // streaming array2d adds 2 more lines: 1 for dimension + 1 return:
  EXPECT_EQ(int(lines.size()), (M+2)*size);
  int j=0;
  for (int i=0; i<size;i++) {
    EXPECT_EQ(lines[j++], "2 3"); //matrix dimension
    for (int row=0; row<M; row++) {
      ostringstream expected;
      for (int n=0; n<N; n++) expected << inputData[i][row][n] << " ";
      EXPECT_EQ(lines[j++], expected.str());
    }
    EXPECT_EQ(lines[j++], "");
  }
}

TEST(FileOutput, DifferentTypes) {
  string filename = "build/test/fileoutput.txt";
  const char* inputData[] = {"foo", "bar"};

  streaming::Algorithm* gen = new VectorInput<string>(inputData);
  streaming::Algorithm* file = new FileOutput<Real>();
  file->configure("filename", filename, "mode", "text");

  EXPECT_THROW(connect(gen->output("data"), file->input("data")), EssentiaException);

  delete file;
  delete gen;
}

TEST(FileOutput, DifferentSizes) {
  string filename = "build/test/fileoutput.txt";
  const char* inputData0[] = {"foo0", "foo01"};
  const char* inputData1[] = {"foo1", "foo11, foo12"};
  vector<vector<string> > inputData;
  inputData.push_back(arrayToVector<string>(inputData0));
  inputData.push_back(arrayToVector<string>(inputData1));

  streaming::Algorithm* gen = new VectorInput<vector<string> >(&inputData);
  streaming::Algorithm* file = new FileOutput<vector<string> >();
  file->configure("filename", filename, "mode", "text");

  connect(gen->output("data"), file->input("data"));

  scheduler::Network(gen).run();

  ifstream fileStream;
  fileStream.open(filename.c_str());
  vector<string> lines;
  string line;
  while (getline(fileStream, line)) lines.push_back(line);

  fileStream.close();
  if (remove(filename.c_str())) throw EssentiaException("TestFileOutput: Error deleting ", filename);

  EXPECT_EQ(int(lines.size()), int(inputData.size()));
  EXPECT_EQ(lines[0], "[foo0, foo01]");
  EXPECT_EQ(lines[1], "[foo1, foo11, foo12]");
}

TEST(FileOutput, InvalidParam) {
  string filename =  "build/test/invalidParam.txt";
  streaming::Algorithm* file = new FileOutput<vector<complex<Real> > >();
  string expected = "Parameter mode=\"unknown\" is not within specified range: {text,binary}";
  ASSERT_THROW(file->configure("filename", filename, "mode", "unknown"),
               EssentiaException /*, e.what(), expected */);

  filename = "";
  expected = "FileOutput: empty filenames are not allowed.";
  ASSERT_THROW(file->configure("filename", filename, "mode", "text"),
               EssentiaException /* , e.what(), expected */);

  delete file;
}
