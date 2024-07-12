/*
 * Copyright (C) 2006-2021  Music Technology Group - Universitat Pompeu Fabra
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
#include "directscaletransform.h"
#include "essentiamath.h"

#define FOR(i,l,r) for(int i=l; i<r; i++)

using namespace essentia;
using namespace standard;

const char* DirectScaleTransform::name = "DirectScaleTransform";
const char* DirectScaleTransform::category = "Standard";
const char* DirectScaleTransform::description = 
DOC("Computes Direct Scale Transform on a given matrix.\n" 
    "This code was derived from the original DST paper:\n"
    "[1] Williams, W. J., and E. J. Zalubas. Helicopter transmission fault detection via "
    "time-frequency, scale and spectral methods. Mechanical systems and signal processing"
    "14.4 (2000): 545-559.");

void DirectScaleTransform::compute() {

    const std::vector<std::vector<Real>>& matrix = _matrix.get();
    std::vector<std::vector<Real>>& result = _result.get();

    int N = matrix[0].size();
    int C = matrix.size();
    Real step = M_PI/log(N+1);
    int num_rows = C/step;

    result.resize(num_rows, std::vector<Real>(N-1, 0));
    Real Ts = 1/fs;

    FOR(i, 0, num_rows) {
        FOR(j, 0, N-1) {
            Real c = step * i;
            Real k = j + 1;
            complex<Real> k_ = complex<Real>(k * Ts);
            complex<Real> c_ = complex<Real>(0.5) - zi * c; 

            complex<Real> M = pow(k_, c_)/(c_ * sqrt(2*M_PI));
            result[i][j] = M.real();
        }
    }
}

// vector<vector< complex<double> >> DirectScaleTransform(int N=10, int C=6, int fs=1) {
//     complex<double> zi = 1i;
//     double step = M_PI/log(N+1);
//     int num_rows = C/step;

//     vector<vector< complex<double> >> result(num_rows, vector< complex<double> >(N-1, 0));
//     double Ts = 1/fs;

//     FOR(i, 0, num_rows) {
//         FOR(j, 0, N-1) {
//             double c = step * i;
//             double k = j + 1;
//             complex<double> k_ = complex<double>(k * Ts);
//             complex<double> c_ = complex<double>(0.5) - zi * c; 

//             complex<double> M = pow(k_, c_)/(c_ * sqrt(2*M_PI));
//             result[i][j] = M;
//         }
//     }
//     return result;
// }

// int main() {
//     vector<vector< complex<double> >> result = DirectScaleTransform();
//     for(auto row: result){
//         for(auto elem: row){
//             cout << elem.real() << "+" << elem.imag() << "j ";
//         }
//         cout << endl;
//     }
//     return 0;
// }