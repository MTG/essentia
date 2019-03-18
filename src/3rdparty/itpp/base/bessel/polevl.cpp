/*!
 * \file
 * \brief Implementation of polynomial evaluation function
 * \author Tony Ottosson
 *
 * -------------------------------------------------------------------------
 *
 * Copyright (C) 1995-2010  (see AUTHORS file for a list of contributors)
 *
 * This file is part of IT++ - a C++ library of mathematical, signal
 * processing, speech processing, and communications classes and functions.
 *
 * IT++ is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * IT++ is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along
 * with IT++.  If not, see <http://www.gnu.org/licenses/>.
 *
 * -------------------------------------------------------------------------
 *
 * This is slightly modified routine from the Cephes library:
 * http://www.netlib.org/cephes/
 */

#include "bessel_internal.h"


/*
 * Evaluate polynomial
 *
 * int N;
 * double x, y, coef[N+1], polevl[];
 *
 * y = polevl( x, coef, N );
 *
 * DESCRIPTION:
 *
 * Evaluates polynomial of degree N:
 *
 *                     2          N
 * y  =  C  + C x + C x  +...+ C x
 *        0    1     2          N
 *
 * Coefficients are stored in reverse order:
 *
 * coef[0] = C  , ..., coef[N] = C  .
 *            N                   0
 *
 *  The function p1evl() assumes that coef[N] = 1.0 and is
 * omitted from the array.  Its calling arguments are
 * otherwise the same as polevl().
 *
 * SPEED:
 *
 * In the interest of speed, there are no checks for out
 * of bounds arithmetic.  This routine is used by most of
 * the functions in the library.  Depending on available
 * equipment features, the user may wish to rewrite the
 * program in microcode or assembly language.
 */

/*
  Cephes Math Library Release 2.1:  December, 1988
  Copyright 1984, 1987, 1988 by Stephen L. Moshier
*/

double polevl(double x, double coef[], int N)
{
  double ans;
  int i;
  double *p;

  p = coef;
  ans = *p++;
  i = N;

  do
    ans = ans * x  +  *p++;
  while (--i);

  return(ans);
}

/*                                          N
 * Evaluate polynomial when coefficient of x  is 1.0.
 * Otherwise same as polevl.
 */

double p1evl(double x, double coef[], int N)
{
  double ans;
  double *p;
  int i;

  p = coef;
  ans = x + *p++;
  i = N - 1;

  do
    ans = ans * x  + *p++;
  while (--i);

  return(ans);
}
