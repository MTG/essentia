/*							gamma.c
 *
 *	Gamma function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, gamma();
 * extern int sgngam;
 *
 * y = gamma( x );
 *
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
 *    DEC      -34, 34      10000       1.3e-16     2.5e-17
 *    IEEE    -170,-33      20000       2.3e-15     3.3e-16
 *    IEEE     -33,  33     20000       9.4e-16     2.2e-16
 *    IEEE      33, 171.6   20000       2.3e-15     3.2e-16
 *
 * Error for arguments outside the test range will be larger
 * owing to error amplification by the exponential function.
 *
 */
/*							lgam()
 *
 *	Natural logarithm of gamma function
 *
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
 * Arguments greater than MAXLGM return MAXNUM and an error
 * message.  MAXLGM = 2.035093e36 for DEC
 * arithmetic or 2.556348e305 for IEEE arithmetic.
 *
 *
 *
 * ACCURACY:
 *
 *
 * arithmetic      domain        # trials     peak         rms
 *    DEC     0, 3                  7000     5.2e-17     1.3e-17
 *    DEC     2.718, 2.035e36       5000     3.9e-17     9.9e-18
 *    IEEE    0, 3                 28000     5.4e-16     1.1e-16
 *    IEEE    2.718, 2.556e305     40000     3.5e-16     8.3e-17
 * The error criterion was relative when the function magnitude
 * was greater than one but absolute when it was less than one.
 *
 * The following test used the relative error criterion, though
 * at certain points the relative error could be much higher than
 * indicated.
 *    IEEE    -200, -4             10000     4.8e-16     1.3e-16
 *
 */

/*							gamma.c	*/
/*	gamma function	*/

/*
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier
*/


#include "bessel.h"

namespace cephes{

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

// #ifdef ANSIPROT
// extern double pow ( double, double );
// extern double log ( double );
// extern double exp ( double );
// extern double sin ( double );
// extern double polevl ( double, void *, int );
// extern double p1evl ( double, void *, int );
// extern double floor ( double );
// extern double fabs ( double );
// extern int isnan ( double );
// extern int isfinite ( double );
// static double stirf ( double );
// double lgam ( double );
// #else
// double pow(), log(), exp(), sin(), polevl(), p1evl(), floor(), fabs();
// int isnan(), isfinite();
// static double stirf();
// double lgam();
// #endif
// #ifdef INFINITIES
// extern double INFINITY;
// #endif
// #ifdef NANS
// extern double NAN;
// #endif


/* Gamma function computed by Stirling's formula.
 * The polynomial STIR is valid for 33 <= x <= 172.
 */
static double stirf(double x)
{
double y, w, v;

w = 1.0/x;
w = 1.0 + w * polevl( w, STIR, 4 );
y = exp(x);
if( x > MAXSTIR )
	{ /* Avoid overflow in pow() */
	v = pow( x, 0.5 * x - 0.25 );
	y = v * (v / y);
	}
else
	{
	y = pow( x, x - 0.5 ) / y;
	}
y = SQTPI * y * w;
return( y );
}



double gamma(double x)
{
double p, q, z;
int i;

sgngam = 1;
#ifdef NANS
if( isnan(x) )
	return(x);
#endif
#ifdef INFINITIES
#ifdef NANS
if( x == INFINITY )
	return(x);
if( x == -INFINITY )
	return(NAN);
#else
if( !isfinite(x) )
	return(x);
#endif
#endif
q = fabs(x);

if( q > 33.0 )
	{
	if( x < 0.0 )
		{
		p = floor(q);
		if( p == q )
			{
#ifdef NANS
gamnan:
			// mtherr( "gamma", DOMAIN );
			return (NAN);
#else
			goto goverf;
#endif
			}
		i = p;
		if( (i & 1) == 0 )
			sgngam = -1;
		z = q - p;
		if( z > 0.5 )
			{
			p += 1.0;
			z = q - p;
			}
		z = q * sin( PI * z );
		if( z == 0.0 )
			{
#ifdef INFINITIES
			return( sgngam * INFINITY);
#else
goverf:
			// mtherr( "gamma", OVERFLOW );
			return( sgngam * MAXNUM);
#endif
			}
		z = fabs(z);
		z = PI/(z * stirf(q) );
		}
	else
		{
		z = stirf(x);
		}
	return( sgngam * z );
	}

z = 1.0;
while( x >= 3.0 )
	{
	x -= 1.0;
	z *= x;
	}

while( x < 0.0 )
	{
	if( x > -1.E-9 )
		goto small;
	z /= x;
	x += 1.0;
	}

while( x < 2.0 )
	{
	if( x < 1.e-9 )
		goto small;
	z /= x;
	x += 1.0;
	}

if( x == 2.0 )
	return(z);

x -= 2.0;
p = polevl( x, P, 6 );
q = polevl( x, Q, 7 );
return( z * p / q );

small:
if( x == 0.0 )
	{
#ifdef INFINITIES
#ifdef NANS
	  goto gamnan;
#else
	  return( INFINITY );
#endif
#else
	// mtherr( "gamma", SING );
	return( MAXNUM );
#endif
	}
else
	return( z/((1.0 + 0.5772156649015329 * x) * x) );
}



/* A[]: Stirling's formula expansion of log gamma
 * B[], C[]: log gamma function between 2 and 3
 */
#ifdef UNK
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
#define MAXLGM 2.556348e305
#endif

#ifdef DEC
static unsigned short A[] = {
0035524,0141201,0034633,0031405,
0135433,0176755,0126007,0045030,
0035520,0006371,0003342,0172730,
0136066,0005540,0132605,0026407,
0037252,0125252,0125252,0125132
};
static unsigned short B[] = {
0142654,0044014,0077633,0035410,
0144027,0110641,0125335,0144760,
0144641,0165637,0142204,0047447,
0145215,0162027,0146246,0155211,
0145322,0026110,0010317,0110130,
0145120,0061472,0120300,0025363
};
static unsigned short C[] = {
/*0040200,0000000,0000000,0000000*/
0142257,0164150,0163630,0112622,
0143605,0050153,0156116,0135272,
0144527,0056045,0145642,0062332,
0145213,0012063,0106250,0001025,
0145432,0111254,0044577,0115142,
0145366,0071133,0050217,0005122
};
/* log( sqrt( 2*pi ) ) */
static unsigned short LS2P[] = {040153,037616,041445,0172645,};
#define LS2PI *(double *)LS2P
#define MAXLGM 2.035093e36
#endif

#ifdef IBMPC
static unsigned short A[] = {
0x6661,0x2733,0x9850,0x3f4a,
0xe943,0xb580,0x7fbd,0xbf43,
0x5ebb,0x20dc,0x019f,0x3f4a,
0xa5a1,0x16b0,0xc16c,0xbf66,
0x554b,0x5555,0x5555,0x3fb5
};
static unsigned short B[] = {
0x6761,0x8ff3,0x8901,0xc095,
0xb93e,0x355b,0xf234,0xc0e2,
0x89e5,0xf890,0x3d73,0xc114,
0xdb51,0xf994,0xbc82,0xc131,
0xf20b,0x0219,0x4589,0xc13a,
0x055e,0x5418,0x0c67,0xc12a
};
static unsigned short C[] = {
/*0x0000,0x0000,0x0000,0x3ff0,*/
0x12b2,0x1cf3,0xfd0d,0xc075,
0xd757,0x7b89,0xaa0d,0xc0d0,
0x4c9b,0xb974,0xeb84,0xc10a,
0x0043,0x7195,0x6286,0xc131,
0xf34c,0x892f,0x5255,0xc143,
0xe14a,0x6a11,0xce4b,0xc13e
};
/* log( sqrt( 2*pi ) ) */
static unsigned short LS2P[] = {
0xbeb5,0xc864,0x67f1,0x3fed
};
#define LS2PI *(double *)LS2P
#define MAXLGM 2.556348e305
#endif

#ifdef MIEEE
static unsigned short A[] = {
0x3f4a,0x9850,0x2733,0x6661,
0xbf43,0x7fbd,0xb580,0xe943,
0x3f4a,0x019f,0x20dc,0x5ebb,
0xbf66,0xc16c,0x16b0,0xa5a1,
0x3fb5,0x5555,0x5555,0x554b
};
static unsigned short B[] = {
0xc095,0x8901,0x8ff3,0x6761,
0xc0e2,0xf234,0x355b,0xb93e,
0xc114,0x3d73,0xf890,0x89e5,
0xc131,0xbc82,0xf994,0xdb51,
0xc13a,0x4589,0x0219,0xf20b,
0xc12a,0x0c67,0x5418,0x055e
};
static unsigned short C[] = {
0xc075,0xfd0d,0x1cf3,0x12b2,
0xc0d0,0xaa0d,0x7b89,0xd757,
0xc10a,0xeb84,0xb974,0x4c9b,
0xc131,0x6286,0x7195,0x0043,
0xc143,0x5255,0x892f,0xf34c,
0xc13e,0xce4b,0x6a11,0xe14a
};
/* log( sqrt( 2*pi ) ) */
static unsigned short LS2P[] = {
0x3fed,0x67f1,0xc864,0xbeb5
};
#define LS2PI *(double *)LS2P
#define MAXLGM 2.556348e305
#endif

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

/* Logarithm of gamma function */


double lgam(double x)
{
double p, q, u, w, z;
int i;

sgngam = 1;
#ifdef NANS
if( isnan(x) )
	return(x);
#endif

#ifdef INFINITIES
if( !isfinite(x) )
	return(INFINITY);
#endif

if( x < -34.0 )
	{
	q = -x;
	w = lgam(q); /* note this modifies sgngam! */
	p = floor(q);
	if( p == q )
		{
lgsing:
#ifdef INFINITIES
		// mtherr( "lgam", SING );
		return (INFINITY);
#else
		goto loverf;
#endif
		}
	i = p;
	if( (i & 1) == 0 )
		sgngam = -1;
	else
		sgngam = 1;
	z = q - p;
	if( z > 0.5 )
		{
		p += 1.0;
		z = p - q;
		}
	z = q * sin( PI * z );
	if( z == 0.0 )
		goto lgsing;
/*	z = log(PI) - log( z ) - w;*/
	z = LOGPI - log( z ) - w;
	return( z );
	}

if( x < 13.0 )
	{
	z = 1.0;
	p = 0.0;
	u = x;
	while( u >= 3.0 )
		{
		p -= 1.0;
		u = x + p;
		z *= u;
		}
	while( u < 2.0 )
		{
		if( u == 0.0 )
			goto lgsing;
		z /= u;
		p += 1.0;
		u = x + p;
		}
	if( z < 0.0 )
		{
		sgngam = -1;
		z = -z;
		}
	else
		sgngam = 1;
	if( u == 2.0 )
		return( log(z) );
	p -= 2.0;
	x = x + p;
	p = x * polevl( x, B, 5 ) / p1evl( x, C, 6);
	return( log(z) + p );
	}

if( x > MAXLGM )
	{
#ifdef INFINITIES
	return( sgngam * INFINITY );
#else
loverf:
	// mtherr( "lgam", OVERFLOW );
	return( sgngam * MAXNUM );
#endif
	}

q = ( x - 0.5 ) * log(x) - x + LS2PI;
if( x > 1.0e8 )
	return( q );

p = 1.0/(x*x);
if( x >= 1000.0 )
	q += ((   7.9365079365079365079365e-4 * p
		- 2.7777777777777777777778e-3) *p
		+ 0.0833333333333333333333) / x;
else
	q += polevl( p, A, 4 ) / x;
return( q );
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
	// These lines do not make sense
	// therefore we comment them to 
	// prevent a compilation warning (Pablo A.)
	// if (std::isinf(x) == -1)
	// return(NAN);

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

} // namespace cephes
