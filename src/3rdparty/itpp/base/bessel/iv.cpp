/*!
 * \file
 * \brief Implementation of modified Bessel functions of noninteager order
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
// #include <itpp/base/itassert.h>


/*
 * Modified Bessel function of noninteger order
 *
 * double v, x, y, iv();
 *
 * y = iv( v, x );
 *
 * DESCRIPTION:
 *
 * Returns modified Bessel function of order v of the
 * argument.  If x is negative, v must be integer valued.
 *
 * The function is defined as Iv(x) = Jv( ix ).  It is
 * here computed in terms of the confluent hypergeometric
 * function, according to the formula
 *
 *              v  -x
 * Iv(x) = (x/2)  e   hyperg( v+0.5, 2v+1, 2x ) / gamma(v+1)
 *
 * If v is a negative integer, then v is replaced by -v.
 *
 *
 * ACCURACY:
 *
 * Tested at random points (v, x), with v between 0 and
 * 30, x between 0 and 28.
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0,30         10000      1.7e-14     2.7e-15
 *
 * Accuracy is diminished if v is near a negative integer.
 *
 * See also hyperg.c.
 */

/* Mdified Bessel function of noninteger order */
/* If x < 0, then v must be an integer. */

/*
  Cephes Math Library Release 2.8:  June, 2000
  Copyright 1984, 1987, 1988, 2000 by Stephen L. Moshier
*/


#define MAXNUM 1.79769313486231570815E308    /* 2**1024*(1-MACHEP) */

double iv(double v, double x)
{
  int sign;
  double t, ax;

  /* If v is a negative integer, invoke symmetry */
  t = floor(v);
  if (v < 0.0) {
    if (t == v) {
      v = -v; /* symmetry */
      t = -t;
    }
  }
  /* If x is negative, require v to be an integer */
  sign = 1;
  if (x < 0.0) {
    if (t != v) {
      // it_warning("iv(): argument domain error");
      return(0.0);
    }
    if (v != 2.0 * floor(v / 2.0))
      sign = -1;
  }

  /* Avoid logarithm singularity */
  if (x == 0.0) {
    if (v == 0.0)
      return(1.0);
    if (v < 0.0) {
      // it_warning("iv(): overflow range error");
      return(MAXNUM);
    }
    else
      return(0.0);
  }

  ax = fabs(x);
  t = v * log(0.5 * ax)  -  x;
  t = sign * exp(t) / gam(v + 1.0);
  ax = v + 0.5;
  return(t * hyperg(ax,  2.0 * ax,  2.0 * x));
}
