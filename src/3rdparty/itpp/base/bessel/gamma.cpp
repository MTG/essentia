/*!
 * \file
 * \brief Implementation of gamma functions
 * \author Adam Piatyszek
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
// #include <itpp/base/bessel/bessel_internal.h>
// #include <itpp/base/itassert.h>
// #include <itpp/base/itcompat.h>
// #include <itpp/base/math/misc.h>


/*
 * Gamma function
 *
 *
 * SYNOPSIS:
 *
 * double x, y, gam();
 * extern int sgngam;
 *
 * y = gam( x );
 *
 *
 * DESCRIPTION:
 *
 * Returns gamma function of the argument.  The result is
 * correctly signed, and the sign (+1 or -1) is also
 * returned in a global (extern) variable named sgngam.
 * This variable is also filled in by the logarithmic gamma
 * function lgam().
 *
 * Arguments |x| <= 34 are reduced by recurrence and the function
 * approximated by a rational function of degree 6/7 in the
 * interval (2,3).  Large arguments are handled by Stirling's
 * formula. Large negative arguments are made positive using
 * a reflection formula.
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE    -170,-33      20000       2.3e-15     3.3e-16
 *    IEEE     -33,  33     20000       9.4e-16     2.2e-16
 *    IEEE      33, 171.6   20000       2.3e-15     3.2e-16
 *
 * Error for arguments outside the test range will be larger
 * owing to error amplification by the exponential function.
 */

/*
 * Natural logarithm of gamma function
 *
 *
 * SYNOPSIS:
 *
 * double x, y, lgam();
 * extern int sgngam;
 *
 * y = lgam( x );
 *
 *
 * DESCRIPTION:
 *
 * Returns the base e (2.718...) logarithm of the absolute
 * value of the gamma function of the argument.
 * The sign (+1 or -1) of the gamma function is returned in a
 * global (extern) variable named sgngam.
 *
 * For arguments greater than 13, the logarithm of the gamma
 * function is approximated by the logarithmic version of
 * Stirling's formula using a polynomial approximation of
 * degree 4. Arguments between -33 and +33 are reduced by
 * recurrence to the interval [2,3] of a rational approximation.
 * The cosecant reflection formula is employed for arguments
 * less than -33.
 *
 * Arguments greater than MAXLGM return INFINITY and an error
 * message.  MAXLGM = 2.556348e305 for IEEE arithmetic.
 *
 *
 * ACCURACY:
 *
 * arithmetic      domain        # trials     peak         rms
 *    IEEE    0, 3                 28000     5.4e-16     1.1e-16
 *    IEEE    2.718, 2.556e305     40000     3.5e-16     8.3e-17
 * The error criterion was relative when the function magnitude
 * was greater than one but absolute when it was less than one.
 *
 * The following test used the relative error criterion, though
 * at certain points the relative error could be much higher than
 * indicated.
 *    IEEE    -200, -4             10000     4.8e-16     1.3e-16
 */

/*
  Cephes Math Library Release 2.8:  June, 2000
  Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier
*/

#ifndef INFINITY
#  define INFINITY 1.79769313486231570815E308 /* 2**1024*(1-MACHEP) */
#endif
#ifndef NAN
#  define NAN 0.0
#endif


static double P[] = {
  1.60119522476751861407E-4,
  1.19135147006586384913E-3,
  1.04213797561761569935E-2,
  4.76367800457137231464E-2,
  2.07448227648435975150E-1,
  4.94214826801497100753E-1,
  9.99999999999999996796E-1
};
static double Q[] = {
  -2.31581873324120129819E-5,
  5.39605580493303397842E-4,
  -4.45641913851797240494E-3,
  1.18139785222060435552E-2,
  3.58236398605498653373E-2,
  -2.34591795718243348568E-1,
  7.14304917030273074085E-2,
  1.00000000000000000320E0
};
static double LOGPI = 1.14472988584940017414;
static double PI = 3.14159265358979323846;       /* pi */

/* Stirling's formula for the gamma function */
static double STIR[5] = {
  7.87311395793093628397E-4,
  -2.29549961613378126380E-4,
  -2.68132617805781232825E-3,
  3.47222221605458667310E-3,
  8.33333333333482257126E-2,
};
static double MAXSTIR = 143.01608;
static double SQTPI = 2.50662827463100050242E0;
static double MAXLGM = 2.556348e305;

int sgngam = 0;

/*!
 * \brief Gamma function computed by Stirling's formula.
 * The polynomial STIR is valid for 33 <= x <= 172.
 */
static double stirf(double x)
{
  double y, w, v;

  w = 1.0 / x;
  w = 1.0 + w * polevl(w, STIR, 4);
  y = exp(x);
  if (x > MAXSTIR) { /* Avoid overflow in pow() */
    v = pow(x, 0.5 * x - 0.25);
    y = v * (v / y);
  }
  else {
    y = pow(x, x - 0.5) / y;
  }
  y = SQTPI * y * w;
  return(y);
}



double gam(double x)
{
  double p, q, z;
  int i;

  sgngam = 1;
  if (std::isnan(x))
    return(x);

  if (std::isinf(x) == 1)
    return(x);
  if (std::isinf(x) == -1)
    return(NAN);

  q = fabs(x);

  if (q > 33.0) {
    if (x < 0.0) {
      p = floor(q);
      if (p == q) {
      gamnan:
        // it_warning("gam(): argument domain error");
        return (NAN);
      }
      i = int(p);
      if ((i & 1) == 0)
        sgngam = -1;
      z = q - p;
      if (z > 0.5) {
        p += 1.0;
        z = q - p;
      }
      z = q * sin(PI * z);
      if (z == 0.0) {
        return(sgngam * INFINITY);
      }
      z = fabs(z);
      z = PI / (z * stirf(q));
    }
    else {
      z = stirf(x);
    }
    return(sgngam * z);
  }

  z = 1.0;
  while (x >= 3.0) {
    x -= 1.0;
    z *= x;
  }

  while (x < 0.0) {
    if (x > -1.E-9)
      goto small;
    z /= x;
    x += 1.0;
  }

  while (x < 2.0) {
    if (x < 1.e-9)
      goto small;
    z /= x;
    x += 1.0;
  }

  if (x == 2.0)
    return(z);

  x -= 2.0;
  p = polevl(x, P, 6);
  q = polevl(x, Q, 7);
  return(z * p / q);

small:
  if (x == 0.0) {
    goto gamnan;
  }
  else
    return(z / ((1.0 + 0.5772156649015329 * x) * x));
}



/* A[]: Stirling's formula expansion of log gamma
 * B[], C[]: log gamma function between 2 and 3
 */
static double A[] = {
  8.11614167470508450300E-4,
  -5.95061904284301438324E-4,
  7.93650340457716943945E-4,
  -2.77777777730099687205E-3,
  8.33333333333331927722E-2
};
static double B[] = {
  -1.37825152569120859100E3,
  -3.88016315134637840924E4,
  -3.31612992738871184744E5,
  -1.16237097492762307383E6,
  -1.72173700820839662146E6,
  -8.53555664245765465627E5
};
static double C[] = {
  /* 1.00000000000000000000E0, */
  -3.51815701436523470549E2,
  -1.70642106651881159223E4,
  -2.20528590553854454839E5,
  -1.13933444367982507207E6,
  -2.53252307177582951285E6,
  -2.01889141433532773231E6
};
/* log( sqrt( 2*pi ) ) */
static double LS2PI  =  0.91893853320467274178;


/*! Logarithm of gamma function */
double lgam(double x)
{
  double p, q, u, w, z;
  int i;

  sgngam = 1;
  if (std::isnan(x))
    return(x);

  if (!std::isfinite(x))
    return(INFINITY);

  if (x < -34.0) {
    q = -x;
    w = lgam(q); /* note this modifies sgngam! */
    p = floor(q);
    if (p == q) {
    lgsing:
      // it_warning("lgam(): function singularity");
      return (INFINITY);
    }
    i = int(p);
    if ((i & 1) == 0)
      sgngam = -1;
    else
      sgngam = 1;
    z = q - p;
    if (z > 0.5) {
      p += 1.0;
      z = p - q;
    }
    z = q * sin(PI * z);
    if (z == 0.0)
      goto lgsing;
    /*      z = log(PI) - log( z ) - w;*/
    z = LOGPI - log(z) - w;
    return(z);
  }

  if (x < 13.0) {
    z = 1.0;
    p = 0.0;
    u = x;
    while (u >= 3.0) {
      p -= 1.0;
      u = x + p;
      z *= u;
    }
    while (u < 2.0) {
      if (u == 0.0)
        goto lgsing;
      z /= u;
      p += 1.0;
      u = x + p;
    }
    if (z < 0.0) {
      sgngam = -1;
      z = -z;
    }
    else
      sgngam = 1;
    if (u == 2.0)
      return(log(z));
    p -= 2.0;
    x = x + p;
    p = x * polevl(x, B, 5) / p1evl(x, C, 6);
    return(log(z) + p);
  }

  if (x > MAXLGM) {
    return(sgngam * INFINITY);
  }

  q = (x - 0.5) * log(x) - x + LS2PI;
  if (x > 1.0e8)
    return(q);

  p = 1.0 / (x * x);
  if (x >= 1000.0)
    q += ((7.9365079365079365079365e-4 * p
           - 2.7777777777777777777778e-3) * p
          + 0.0833333333333333333333) / x;
  else
    q += polevl(p, A, 4) / x;
  return(q);
}
