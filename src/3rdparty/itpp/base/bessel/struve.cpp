/*!
 * \file
 * \brief Implementation of struve function
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

#include <itpp/base/bessel/bessel_internal.h>
#include <itpp/base/itcompat.h>

/*
 * Struve function
 *
 * double v, x, y, struve();
 *
 * y = struve( v, x );
 *
 * DESCRIPTION:
 *
 * Computes the Struve function Hv(x) of order v, argument x.
 * Negative x is rejected unless v is an integer.
 *
 * This module also contains the hypergeometric functions 1F2
 * and 3F0 and a routine for the Bessel function Yv(x) with
 * noninteger v.
 *
 * ACCURACY:
 *
 * Not accurately characterized, but spot checked against tables.
 */

/*
  Cephes Math Library Release 2.81:  June, 2000
  Copyright 1984, 1987, 1989, 2000 by Stephen L. Moshier
*/

static double stop = 1.37e-17;

#define MACHEP 1.11022302462515654042E-16   /* 2**-53 */

double onef2(double a, double b, double c, double x, double *err)
{
  double n, a0, sum, t;
  double an, bn, cn, max, z;

  an = a;
  bn = b;
  cn = c;
  a0 = 1.0;
  sum = 1.0;
  n = 1.0;
  t = 1.0;
  max = 0.0;

  do {
    if (an == 0)
      goto done;
    if (bn == 0)
      goto error;
    if (cn == 0)
      goto error;
    if ((a0 > 1.0e34) || (n > 200))
      goto error;
    a0 *= (an * x) / (bn * cn * n);
    sum += a0;
    an += 1.0;
    bn += 1.0;
    cn += 1.0;
    n += 1.0;
    z = fabs(a0);
    if (z > max)
      max = z;
    if (sum != 0)
      t = fabs(a0 / sum);
    else
      t = z;
  }
  while (t > stop);

done:

  *err = fabs(MACHEP * max / sum);

  goto xit;

error:
  *err = 1.0e38;

xit:
  return(sum);
}


double threef0(double a, double b, double c, double x, double *err)
{
  double n, a0, sum, t, conv, conv1;
  double an, bn, cn, max, z;

  an = a;
  bn = b;
  cn = c;
  a0 = 1.0;
  sum = 1.0;
  n = 1.0;
  t = 1.0;
  max = 0.0;
  conv = 1.0e38;
  conv1 = conv;

  do {
    if (an == 0.0)
      goto done;
    if (bn == 0.0)
      goto done;
    if (cn == 0.0)
      goto done;
    if ((a0 > 1.0e34) || (n > 200))
      goto error;
    a0 *= (an * bn * cn * x) / n;
    an += 1.0;
    bn += 1.0;
    cn += 1.0;
    n += 1.0;
    z = fabs(a0);
    if (z > max)
      max = z;
    if (z >= conv) {
      if ((z < max) && (z > conv1))
        goto done;
    }
    conv1 = conv;
    conv = z;
    sum += a0;
    if (sum != 0)
      t = fabs(a0 / sum);
    else
      t = z;
  }
  while (t > stop);

done:

  t = fabs(MACHEP * max / sum);

  max = fabs(conv / sum);
  if (max > t)
    t = max;

  goto xit;

error:
  t = 1.0e38;

xit:

  *err = t;
  return(sum);
}

#define PI 3.14159265358979323846       /* pi */

double struve(double v, double x)
{
  double y, ya, f, g, h, t;
  double onef2err, threef0err;

  f = floor(v);
  if ((v < 0) && (v - f == 0.5)) {
    y = jv(-v, x);
    f = 1.0 - f;
    g =  2.0 * floor(f / 2.0);
    if (g != f)
      y = -y;
    return(y);
  }
  t = 0.25 * x * x;
  f = fabs(x);
  g = 1.5 * fabs(v);
  if ((f > 30.0) && (f > g)) {
    onef2err = 1.0e38;
    y = 0.0;
  }
  else {
    y = onef2(1.0, 1.5, 1.5 + v, -t, &onef2err);
  }

  if ((f < 18.0) || (x < 0.0)) {
    threef0err = 1.0e38;
    ya = 0.0;
  }
  else {
    ya = threef0(1.0, 0.5, 0.5 - v, -1.0 / t, &threef0err);
  }

  f = sqrt(PI);
  h = pow(0.5 * x, v - 1.0);

  if (onef2err <= threef0err) {
    g = gam(v + 1.5);
    y = y * h * t / (0.5 * f * g);
    return(y);
  }
  else {
    g = gam(v + 0.5);
    ya = ya * h / (f * g);
    ya = ya + yv(v, x);
    return(ya);
  }
}


/*
 * Bessel function of noninteger order
 */
double yv(double v, double x)
{
  double y, t;
  int n;

  y = floor(v);
  if (y == v) {
    n = int(v);
    y = yn(n, x);
    return(y);
  }
  t = PI * v;
  y = (cos(t) * jv(v, x) - jv(-v, x)) / sin(t);
  return(y);
}

/* Crossover points between ascending series and asymptotic series
 * for Struve function
 *
 *  v  x
 *
 *  0 19.2
 *  1 18.95
 *  2 19.15
 *  3 19.3
 *  5 19.7
 * 10 21.35
 * 20 26.35
 * 30 32.31
 * 40 40.0
 */
