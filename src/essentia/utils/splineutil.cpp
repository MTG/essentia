# include <cstdlib>
# include <iostream>
# include <iomanip>
# include <cmath>
# include <ctime>
# include <string>

using namespace std;

# include "splineutil.h"

//****************************************************************************80

double basis_function_b_val ( double tdata[], double tval )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_FUNCTION_B_VAL evaluates the B spline basis function.
//
//  Discussion:
//
//    The B spline basis function is a piecewise cubic which
//    has the properties that:
//
//    * it equals 2/3 at TDATA(3), 1/6 at TDATA(2) and TDATA(4);
//    * it is 0 for TVAL <= TDATA(1) or TDATA(5) <= TVAL;
//    * it is strictly increasing from TDATA(1) to TDATA(3),
//      and strictly decreasing from TDATA(3) to TDATA(5);
//    * the function and its first two derivatives are continuous
//      at each node TDATA(I).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Alan Davies, Philip Samuels,
//    An Introduction to Computational Geometry for Curves and Surfaces,
//    Clarendon Press, 1996,
//    ISBN: 0-19-851478-6,
//    LC: QA448.D38.
//
//  Parameters:
//
//    Input, double TDATA(5), the nodes associated with the basis function.
//    The entries of TDATA are assumed to be distinct and increasing.
//
//    Input, double TVAL, a point at which the B spline basis function is
//    to be evaluated.
//
//    Output, double BASIS_FUNCTION_B_VAL, the value of the function at TVAL.
//
{
# define NDATA 5

  int left=0;
  int right=0;
  double u=0.0;
  double yval=0.0;

  if ( tval <= tdata[0] || tdata[NDATA-1] <= tval )
  {
    yval = 0.0;
    return yval;
  }
//
//  Find the interval [ TDATA(LEFT), TDATA(RIGHT) ] containing TVAL.
//
  r8vec_bracket ( NDATA, tdata, tval, &left, &right );
//
//  U is the normalized coordinate of TVAL in this interval.
//
  u = ( tval - tdata[left-1] ) / ( tdata[right-1] - tdata[left-1] );
//
//  Now evaluate the function.
//
  if ( tval < tdata[1] )
  {
    yval = pow ( u, 3 ) / 6.0;
  }
  else if ( tval < tdata[2] )
  {
    yval = ( ( (     - 3.0
                 * u + 3.0 )
                 * u + 3.0 )
                 * u + 1.0 ) / 6.0;
  }
  else if ( tval < tdata[3])
  {
    yval = ( ( (     + 3.0
                 * u - 6.0 )
                 * u + 0.0 )
                 * u + 4.0 ) / 6.0;
  }
  else if ( tval < tdata[4] )
  {
    yval = pow ( ( 1.0 - u ), 3 ) / 6.0;
  }

  return yval;

# undef NDATA
}
//****************************************************************************80

double basis_function_beta_val ( double beta1, double beta2, double tdata[],
  double tval )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_FUNCTION_BETA_VAL evaluates the beta spline basis function.
//
//  Discussion:
//
//    With BETA1 = 1 and BETA2 = 0, the beta spline basis function
//    equals the B spline basis function.
//
//    With BETA1 large, and BETA2 = 0, the beta spline basis function
//    skews to the right, that is, its maximum increases, and occurs
//    to the right of the center.
//
//    With BETA1 = 1 and BETA2 large, the beta spline becomes more like
//    a linear basis function; that is, its value in the outer two intervals
//    goes to zero, and its behavior in the inner two intervals approaches
//    a piecewise linear function that is 0 at the second node, 1 at the
//    third, and 0 at the fourth.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Alan Davies, Philip Samuels,
//    An Introduction to Computational Geometry for Curves and Surfaces,
//    Clarendon Press, 1996,
//    ISBN: 0-19-851478-6,
//    LC: QA448.D38.
//
//  Parameters:
//
//    Input, double BETA1, the skew or bias parameter.
//    BETA1 = 1 for no skew or bias.
//
//    Input, double BETA2, the tension parameter.
//    BETA2 = 0 for no tension.
//
//    Input, double TDATA[5], the nodes associated with the basis function.
//    The entries of TDATA are assumed to be distinct and increasing.
//
//    Input, double TVAL, a point at which the B spline basis function is
//    to be evaluated.
//
//    Output, double BASIS_FUNCTION_BETA_VAL, the value of the function at TVAL.
//
{
# define NDATA 5

  double a=0.0;
  double b=0.0;
  double c=0.0;
  double d=0.0;
  int left=0;
  int right=0;
  double u=0.0;
  double yval=0.0;

  if ( tval <= tdata[0] || tdata[NDATA-1] <= tval )
  {
    yval = 0.0;
    return yval;
  }
//
//  Find the interval [ TDATA(LEFT), TDATA(RIGHT) ] containing TVAL.
//
  r8vec_bracket ( NDATA, tdata, tval, &left, &right );
//
//  U is the normalized coordinate of TVAL in this interval.
//
  u = ( tval - tdata[left-1] ) / ( tdata[right-1] - tdata[left-1] );
//
//  Now evaluate the function.
//
  if ( tval < tdata[1] )
  {
    yval = 2.0 * u * u * u;
  }
  else if ( tval < tdata[2] )
  {
    a = beta2 + 4.0 * beta1 + 4.0 * beta1 * beta1
      + 6.0 * ( 1.0 - beta1 * beta1 )
      - 3.0 * ( 2.0 + beta2 + 2.0 * beta1 )
      + 2.0 * ( 1.0 + beta2 + beta1 + beta1 * beta1 );

    b = - 6.0 * ( 1.0 - beta1 * beta1 )
        + 6.0 * ( 2.0 + beta2 + 2.0 * beta1 )
        - 6.0 * ( 1.0 + beta2 + beta1 + beta1 * beta1 );

    c = - 3.0 * ( 2.0 + beta2 + 2.0 * beta1 )
        + 6.0 * ( 1.0 + beta2 + beta1 + beta1 * beta1 );

    d = - 2.0 * ( 1.0 + beta2 + beta1 + beta1 * beta1 );

    yval = a + b * u + c * u * u + d * u * u * u;
  }
  else if ( tval < tdata[3] )
  {
    a = beta2 + 4.0 * beta1 + 4.0 * beta1 * beta1;

    b = - 6.0 * beta1 * ( 1.0 - beta1 * beta1 );

    c = - 3.0 * ( beta2 + 2.0 * beta1 * beta1
      + 2.0 * beta1 * beta1 * beta1 );

    d = 2.0 * ( beta2 + beta1 + beta1 * beta1 + beta1 * beta1 * beta1 );

    yval = a + b * u + c * u * u + d * u * u * u;
  }
  else if ( tval < tdata[4] )
  {
    yval = 2.0 * pow ( beta1 * ( 1.0 - u ), 3 );
  }

  yval = yval / ( 2.0 + beta2 + 4.0 * beta1 + 4.0 * beta1 * beta1
    + 2.0 * beta1 * beta1 * beta1 );

  return yval;
# undef NDATA
}
//****************************************************************************80

double *basis_matrix_b_uni ( )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_B_UNI sets up the uniform B spline basis matrix.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    James Foley, Andries vanDam, Steven Feiner, John Hughes,
//    Computer Graphics, Principles and Practice,
//    Second Edition,
//    Addison Wesley, 1995,
//    ISBN: 0201848406,
//    LC: T385.C5735.
//
//  Parameters:
//
//    Output, double BASIS_MATRIX_B_UNI[4*4], the basis matrix.
//
{
  int i;
  int j;
  double *mbasis;
  double mbasis_save[4*4] = {
    -1.0 / 6.0,
     3.0 / 6.0,
    -3.0 / 6.0,
     1.0 / 6.0,
     3.0 / 6.0,
    -6.0 / 6.0,
     0.0,
     4.0 / 6.0,
    -3.0 / 6.0,
     3.0 / 6.0,
     3.0 / 6.0,
     1.0 / 6.0,
     1.0 / 6.0,
     0.0,
     0.0,
     0.0 };

  mbasis = new double[4*4];

  for ( j = 0; j < 4; j++ )
  {
    for ( i = 0; i < 4; i++ )
    {
      mbasis[i+j*4] = mbasis_save[i+j*4];
    }
  }

  return mbasis;
}
//****************************************************************************80

double *basis_matrix_beta_uni ( double beta1, double beta2 )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_BETA_UNI sets up the uniform beta spline basis matrix.
//
//  Discussion:
//
//    If BETA1 = 1 and BETA2 = 0, then the beta spline reduces to
//    the B spline.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    James Foley, Andries vanDam, Steven Feiner, John Hughes,
//    Computer Graphics, Principles and Practice,
//    Second Edition,
//    Addison Wesley, 1995,
//    ISBN: 0201848406,
//    LC: T385.C5735.
//
//  Parameters:
//
//    Input, double BETA1, the skew or bias parameter.
//    BETA1 = 1 for no skew or bias.
//
//    Input, double BETA2, the tension parameter.
//    BETA2 = 0 for no tension.
//
//    Output, double BASIS_MATRIX_BETA_UNI[4*4], the basis matrix.
//
{
  double delta;
  int i;
  int j;
  double *mbasis;

  mbasis = new double[4*4];

  mbasis[0+0*4] = - 2.0 * beta1 * beta1 * beta1;
  mbasis[0+1*4] =   2.0 * beta2
    + 2.0 * beta1 * ( beta1 * beta1 + beta1 + 1.0 );
  mbasis[0+2*4] = - 2.0 * ( beta2 + beta1 * beta1 + beta1 + 1.0 );
  mbasis[0+3*4] =   2.0;

  mbasis[1+0*4] =   6.0 * beta1 * beta1 * beta1;
  mbasis[1+1*4] = - 3.0 * beta2
    - 6.0 * beta1 * beta1 * ( beta1 + 1.0 );
  mbasis[1+2*4] =   3.0 * beta2 + 6.0 * beta1 * beta1;
  mbasis[1+3*4] =   0.0;

  mbasis[2+0*4] = - 6.0 * beta1 * beta1 * beta1;
  mbasis[2+1*4] =   6.0 * beta1 * ( beta1 - 1.0 ) * ( beta1 + 1.0 );
  mbasis[2+2*4] =   6.0 * beta1;
  mbasis[2+3*4] =   0.0;

  mbasis[3+0*4] =   2.0 * beta1 * beta1 * beta1;
  mbasis[3+1*4] =   4.0 * beta1 * ( beta1 + 1.0 ) + beta2;
  mbasis[3+2*4] =   2.0;
  mbasis[3+3*4] =   0.0;

  delta = ( ( 2.0
    * beta1 + 4.0 )
    * beta1 + 4.0 )
    * beta1 + 2.0 + beta2;

  for ( j = 0; j < 4; j++ )
  {
    for ( i = 0; i < 4; i++ )
    {
      mbasis[i+j*4] = mbasis[i+j*4] / delta;
    }
  }

  return mbasis;
}
//****************************************************************************80

double *basis_matrix_bezier ( )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_BEZIER_UNI sets up the cubic Bezier spline basis matrix.
//
//  Discussion:
//
//    This basis matrix assumes that the data points are stored as
//    ( P1, P2, P3, P4 ).  P1 is the function value at T = 0, while
//    P2 is used to approximate the derivative at T = 0 by
//    dP/dt = 3 * ( P2 - P1 ).  Similarly, P4 is the function value
//    at T = 1, and P3 is used to approximate the derivative at T = 1
//    by dP/dT = 3 * ( P4 - P3 ).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    James Foley, Andries vanDam, Steven Feiner, John Hughes,
//    Computer Graphics, Principles and Practice,
//    Second Edition,
//    Addison Wesley, 1995,
//    ISBN: 0201848406,
//    LC: T385.C5735.
//
//  Parameters:
//
//    Output, double BASIS_MATRIX_BEZIER[4*4], the basis matrix.
//
{
  double *mbasis;

  mbasis = new double[4*4];

  mbasis[0+0*4] = -1.0;
  mbasis[0+1*4] =  3.0;
  mbasis[0+2*4] = -3.0;
  mbasis[0+3*4] =  1.0;

  mbasis[1+0*4] =  3.0;
  mbasis[1+1*4] = -6.0;
  mbasis[1+2*4] =  3.0;
  mbasis[1+3*4] =  0.0;

  mbasis[2+0*4] = -3.0;
  mbasis[2+1*4] =  3.0;
  mbasis[2+2*4] =  0.0;
  mbasis[2+3*4] =  0.0;

  mbasis[3+0*4] =  1.0;
  mbasis[3+1*4] =  0.0;
  mbasis[3+2*4] =  0.0;
  mbasis[3+3*4] =  0.0;

  return mbasis;
}
//****************************************************************************80

double *basis_matrix_hermite ( )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_HERMITE sets up the Hermite spline basis matrix.
//
//  Discussion:
//
//    This basis matrix assumes that the data points are stored as
//    ( P1, P2, P1', P2' ), with P1 and P1' being the data value and
//    the derivative dP/dT at T = 0, while P2 and P2' apply at T = 1.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    James Foley, Andries vanDam, Steven Feiner, John Hughes,
//    Computer Graphics, Principles and Practice,
//    Second Edition,
//    Addison Wesley, 1995,
//    ISBN: 0201848406,
//    LC: T385.C5735.
//
//  Parameters:
//
//    Output, double BASIS_MATRIX_HERMITE[4*4], the basis matrix.
//
{
  double *mbasis;

  mbasis = new double[4*4];

  mbasis[0+0*4] =  2.0;
  mbasis[0+1*4] = -2.0;
  mbasis[0+2*4] =  1.0;
  mbasis[0+3*4] =  1.0;

  mbasis[1+0*4] = -3.0;
  mbasis[1+1*4] =  3.0;
  mbasis[1+2*4] = -2.0;
  mbasis[1+3*4] = -1.0;

  mbasis[2+0*4] =  0.0;
  mbasis[2+1*4] =  0.0;
  mbasis[2+2*4] =  1.0;
  mbasis[2+3*4] =  0.0;

  mbasis[3+0*4] =  1.0;
  mbasis[3+1*4] =  0.0;
  mbasis[3+2*4] =  0.0;
  mbasis[3+3*4] =  0.0;

  return mbasis;
}
//****************************************************************************80

double *basis_matrix_overhauser_nonuni ( double alpha, double beta )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_OVERHAUSER_NONUNI sets the nonuniform Overhauser spline basis matrix.
//
//  Discussion:
//
//    This basis matrix assumes that the data points P1, P2, P3 and
//    P4 are not uniformly spaced in T, and that P2 corresponds to T = 0,
//    and P3 to T = 1.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double ALPHA, BETA.
//    ALPHA = || P2 - P1 || / ( || P3 - P2 || + || P2 - P1 || )
//    BETA  = || P3 - P2 || / ( || P4 - P3 || + || P3 - P2 || ).
//
//    Output, double BASIS_MATRIX_OVERHAUSER_NONUNI[4*4], the basis matrix.
//
{
  double *mbasis;

  mbasis = new double[4*4];

  mbasis[0+0*4] = - ( 1.0 - alpha ) * ( 1.0 - alpha ) / alpha;
  mbasis[0+1*4] =   beta + ( 1.0 - alpha ) / alpha;
  mbasis[0+2*4] =   alpha - 1.0 / ( 1.0 - beta );
  mbasis[0+3*4] =   beta * beta / ( 1.0 - beta );

  mbasis[1+0*4] =   2.0 * ( 1.0 - alpha ) * ( 1.0 - alpha ) / alpha;
  mbasis[1+1*4] = ( - 2.0 * ( 1.0 - alpha ) - alpha * beta ) / alpha;
  mbasis[1+2*4] = ( 2.0 * ( 1.0 - alpha )
    - beta * ( 1.0 - 2.0 * alpha ) ) / ( 1.0 - beta );
  mbasis[1+3*4] = - beta * beta / ( 1.0 - beta );

  mbasis[2+0*4] = - ( 1.0 - alpha ) * ( 1.0 - alpha ) / alpha;
  mbasis[2+1*4] =   ( 1.0 - 2.0 * alpha ) / alpha;
  mbasis[2+2*4] =   alpha;
  mbasis[2+3*4] =   0.0;

  mbasis[3+0*4] =   0.0;
  mbasis[3+1*4] =   1.0;
  mbasis[3+2*4] =   0.0;
  mbasis[3+3*4] =   0.0;

  return mbasis;
}
//****************************************************************************80

double *basis_matrix_overhauser_nul ( double alpha )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_OVERHAUSER_NUL sets the nonuniform left Overhauser spline basis matrix.
//
//  Discussion:
//
//    This basis matrix assumes that the data points P1, P2, and
//    P3 are not uniformly spaced in T, and that P1 corresponds to T = 0,
//    and P2 to T = 1. (???)
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double ALPHA.
//    ALPHA = || P2 - P1 || / ( || P3 - P2 || + || P2 - P1 || )
//
//    Output, double BASIS_MATRIX_OVERHAUSER_NUL[3*3], the basis matrix.
//
{
  double *mbasis;

  mbasis = new double[3*3];

  mbasis[0+0*3] =   1.0 / alpha;
  mbasis[0+1*3] = - 1.0 / ( alpha * ( 1.0 - alpha ) );
  mbasis[0+2*3] =   1.0 / ( 1.0 - alpha );

  mbasis[1+0*3] = - ( 1.0 + alpha ) / alpha;
  mbasis[1+1*3] =   1.0 / ( alpha * ( 1.0 - alpha ) );
  mbasis[1+2*3] = - alpha / ( 1.0 - alpha );

  mbasis[2+0*3] =   1.0;
  mbasis[2+1*3] =   0.0;
  mbasis[2+2*3] =   0.0;

  return mbasis;
}
//****************************************************************************80

double *basis_matrix_overhauser_nur ( double beta )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_OVERHAUSER_NUR sets the nonuniform right Overhauser spline basis matrix.
//
//  Discussion:
//
//    This basis matrix assumes that the data points PN-2, PN-1, and
//    PN are not uniformly spaced in T, and that PN-1 corresponds to T = 0,
//    and PN to T = 1. (???)
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double BETA.
//    BETA = || P(N) - P(N-1) || / ( || P(N) - P(N-1) || + || P(N-1) - P(N-2) || )
//
//    Output, double BASIS_MATRIX_OVERHAUSER_NUR[3*3], the basis matrix.
//
{
  double *mbasis;

  mbasis = new double[3*3];

  mbasis[0+0*3] =   1.0 / beta;
  mbasis[0+1*3] = - 1.0 / ( beta * ( 1.0 - beta ) );
  mbasis[0+2*3] =   1.0 / ( 1.0 - beta );

  mbasis[1+0*3] = - ( 1.0 + beta ) / beta;
  mbasis[1+1*3] =   1.0 / ( beta * ( 1.0 - beta ) );
  mbasis[1+2*3] = - beta / ( 1.0 - beta );

  mbasis[2+0*3] =   1.0;
  mbasis[2+1*3] =   0.0;
  mbasis[2+2*3] =   0.0;

  return mbasis;
}
//****************************************************************************80

double *basis_matrix_overhauser_uni ( void)

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_OVERHAUSER_UNI sets the uniform Overhauser spline basis matrix.
//
//  Discussion:
//
//    This basis matrix assumes that the data points P1, P2, P3 and
//    P4 are uniformly spaced in T, and that P2 corresponds to T = 0,
//    and P3 to T = 1.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    James Foley, Andries vanDam, Steven Feiner, John Hughes,
//    Computer Graphics, Principles and Practice,
//    Second Edition,
//    Addison Wesley, 1995,
//    ISBN: 0201848406,
//    LC: T385.C5735.
//
//  Parameters:
//
//    Output, double BASIS_MATRIX_OVERHASUER_UNI[4*4], the basis matrix.
//
{
  double *mbasis;

  mbasis = new double[4*4];

  mbasis[0+0*4] = - 1.0 / 2.0;
  mbasis[0+1*4] =   3.0 / 2.0;
  mbasis[0+2*4] = - 3.0 / 2.0;
  mbasis[0+3*4] =   1.0 / 2.0;

  mbasis[1+0*4] =   2.0 / 2.0;
  mbasis[1+1*4] = - 5.0 / 2.0;
  mbasis[1+2*4] =   4.0 / 2.0;
  mbasis[1+3*4] = - 1.0 / 2.0;

  mbasis[2+0*4] = - 1.0 / 2.0;
  mbasis[2+1*4] =   0.0;
  mbasis[2+2*4] =   1.0 / 2.0;
  mbasis[2+3*4] =   0.0;

  mbasis[3+0*4] =   0.0;
  mbasis[3+1*4] =   2.0 / 2.0;
  mbasis[3+2*4] =   0.0;
  mbasis[3+3*4] =   0.0;

  return mbasis;
}
//****************************************************************************80

double *basis_matrix_overhauser_uni_l ( )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_OVERHAUSER_UNI_L sets the left uniform Overhauser spline basis matrix.
//
//  Discussion:
//
//    This basis matrix assumes that the data points P1, P2, and P3
//    are not uniformly spaced in T, and that P1 corresponds to T = 0,
//    and P2 to T = 1.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double BASIS_MATRIX_OVERHASUER_UNI_L[3*3], the basis matrix.
//
{
  double *mbasis;

  mbasis = new double[3*3];

  mbasis[0+0*3] =   2.0;
  mbasis[0+1*3] = - 4.0;
  mbasis[0+2*3] =   2.0;

  mbasis[1+0*3] = - 3.0;
  mbasis[1+1*3] =   4.0;
  mbasis[1+2*3] = - 1.0;

  mbasis[2+0*3] =   1.0;
  mbasis[2+1*3] =   0.0;
  mbasis[2+2*3] =   0.0;

  return mbasis;
}
//****************************************************************************80

double *basis_matrix_overhauser_uni_r ( )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_OVERHAUSER_UNI_R sets the right uniform Overhauser spline basis matrix.
//
//  Discussion:
//
//    This basis matrix assumes that the data points P(N-2), P(N-1),
//    and P(N) are uniformly spaced in T, and that P(N-1) corresponds to
//    T = 0, and P(N) to T = 1.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Output, double BASIS_MATRIX_OVERHASUER_UNI_R[3*3], the basis matrix.
//
{
  double *mbasis;

  mbasis = new double[3*3];

  mbasis[0+0*3] =   2.0;
  mbasis[0+1*3] = - 4.0;
  mbasis[0+2*3] =   2.0;

  mbasis[1+0*3] = - 3.0;
  mbasis[1+1*3] =   4.0;
  mbasis[1+2*3] = - 1.0;

  mbasis[2+0*3] =   1.0;
  mbasis[2+1*3] =   0.0;
  mbasis[2+2*3] =   0.0;

  return mbasis;
}
//****************************************************************************80

double basis_matrix_tmp ( int left, int n, double mbasis[], int ndata,
  double tdata[], double ydata[], double tval )

//****************************************************************************80
//
//  Purpose:
//
//    BASIS_MATRIX_TMP computes Q = T * MBASIS * P
//
//  Discussion:
//
//    YDATA is a vector of data values, most frequently the values of some
//    function sampled at uniformly spaced points.  MBASIS is the basis
//    matrix for a particular kind of spline.  T is a vector of the
//    powers of the normalized difference between TVAL and the left
//    endpoint of the interval.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int LEFT, indicats that TVAL is in the interval
//    [ TDATA(LEFT), TDATA(LEFT+1) ], or that this is the "nearest"
//    interval to TVAL.
//    For TVAL < TDATA(1), use LEFT = 1.
//    For TDATA(NDATA) < TVAL, use LEFT = NDATA - 1.
//
//    Input, int N, the order of the basis matrix.
//
//    Input, double MBASIS[N*N], the basis matrix.
//
//    Input, int NDATA, the dimension of the vectors TDATA and YDATA.
//
//    Input, double TDATA[NDATA], the abscissa values.  This routine
//    assumes that the TDATA values are uniformly spaced, with an
//    increment of 1.0.
//
//    Input, double YDATA[NDATA], the data values to be interpolated or
//    approximated.
//
//    Input, double TVAL, the value of T at which the spline is to be
//    evaluated.
//
//    Output, double BASIS_MATRIX_TMP, the value of the spline at TVAL.
//
{
  double arg=0.0;
  int first=0;
  int i=0;
  int j=0;
  //double temp;
  double tm=0.0;
  double *tvec=0;
  double yval=0.0;

  tvec = new double[n];

  if ( left == 1 )
  {
    arg = 0.5 * ( tval - tdata[left-1] );
    first = left;
  }
  else if ( left < ndata - 1 )
  {
    arg = tval - tdata[left-1];
    first = left - 1;
  }
  else if ( left == ndata - 1 )
  {
    arg = 0.5 * ( 1.0 + tval - tdata[left-1] );
    first = left - 1;
  }
//
//  TVEC(I) = ARG**(N-I).
//
  tvec[n-1] = 1.0;
  for ( i = n-2; 0 <= i; i-- )
  {
    tvec[i] = arg * tvec[i+1];
  }

  yval = 0.0;
  for ( j = 0; j < n; j++ )
  {
    tm = 0.0;
    for ( i = 0; i < n; i++ )
    {
      tm = tm + tvec[i] * mbasis[i+j*n];
    }
    yval = yval + tm * ydata[first - 1 + j];
  }

  delete [] tvec;

  return yval;
}
//****************************************************************************80

void bc_val ( int n, double t, double xcon[], double ycon[], double *xval,
  double *yval )

//****************************************************************************80
//
//  Purpose:
//
//    BC_VAL evaluates a parameterized Bezier curve.
//
//  Discussion:
//
//    BC_VAL(T) is the value of a vector function of the form
//
//      BC_VAL(T) = ( X(T), Y(T) )
//
//    where
//
//      X(T) = Sum ( 0 <= I <= N ) XCON(I) * BERN(I,N)(T)
//      Y(T) = Sum ( 0 <= I <= N ) YCON(I) * BERN(I,N)(T)
//
//    BERN(I,N)(T) is the I-th Bernstein polynomial of order N
//    defined on the interval [0,1],
//
//    XCON(0:N) and YCON(0:N) are the coordinates of N+1 "control points".
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Kahaner, Cleve Moler, Steven Nash,
//    Numerical Methods and Software,
//    Prentice Hall, 1989,
//    ISBN: 0-13-627258-4,
//    LC: TA345.K34.
//
//  Parameters:
//
//    Input, int N, the order of the Bezier curve, which
//    must be at least 0.
//
//    Input, double T, the point at which the Bezier curve should
//    be evaluated.  The best results are obtained within the interval
//    [0,1] but T may be anywhere.
//
//    Input, double XCON[0:N], YCON[0:N], the X and Y coordinates
//    of the control points.  The Bezier curve will pass through
//    the points ( XCON(0), YCON(0) ) and ( XCON(N), YCON(N) ), but
//    generally NOT through the other control points.
//
//    Output, double *XVAL, *YVAL, the X and Y coordinates of the point
//    on the Bezier curve corresponding to the given T value.
//
{
  double *bval;
  int i;

  bval = bp01 ( n, t );

  *xval = 0.0;
  for ( i = 0; i <= n; i++ )
  {
    *xval = *xval + xcon[i] * bval[i];
  }

  *yval = 0.0;
  for ( i = 0; i <= n; i++ )
  {
    *yval = *yval + ycon[i] * bval[i];
  }

  delete [] bval;

  return;
}
//****************************************************************************80

double bez_val ( int n, double x, double a, double b, double y[] )

//****************************************************************************80
//
//  Purpose:
//
//    BEZ_VAL evaluates a Bezier function at a point.
//
//  Discussion:
//
//    The Bezier function has the form:
//
//      BEZ(X) = Sum ( 0 <= I <= N ) Y(I) * BERN(N,I)( (X-A)/(B-A) )
//
//    BERN(N,I)(X) is the I-th Bernstein polynomial of order N
//    defined on the interval [0,1],
//
//    Y(0:N) is a set of coefficients,
//
//    and if, for I = 0 to N, we define the N+1 points
//
//      X(I) = ( (N-I)*A + I*B) / N,
//
//    equally spaced in [A,B], the pairs ( X(I), Y(I) ) can be regarded as
//    "control points".
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Kahaner, Cleve Moler, Steven Nash,
//    Numerical Methods and Software,
//    Prentice Hall, 1989,
//    ISBN: 0-13-627258-4,
//    LC: TA345.K34.
//
//  Parameters:
//
//    Input, int N, the order of the Bezier function, which
//    must be at least 0.
//
//    Input, double X, the point at which the Bezier function should
//    be evaluated.  The best results are obtained within the interval
//    [A,B] but X may be anywhere.
//
//    Input, double A, B, the interval over which the Bezier function
//    has been defined.  This is the interval in which the control
//    points have been set up.  Note BEZ(A) = Y(0) and BEZ(B) = Y(N),
//    although BEZ will not, in general pass through the other
//    control points.  A and B must not be equal.
//
//    Input, double Y[0:N], a set of data defining the Y coordinates
//    of the control points.
//
//    Output, double BEZ_VAL, the value of the Bezier function at X.
//
{
  double *bval;
  int i;
  double value;
  double x01;

  if ( b - a == 0.0 )
  {
    cout << "\n";
    cout << "BEZ_VAL - Fatal error!\n";
    cout << "  Null interval, A = B = " << a << "\n";
    exit ( 1 );
  }
//
//  X01 lies in [0,1], in the same relative position as X in [A,B].
//
  x01 = ( x - a ) / ( b - a );

  bval = bp01 ( n, x01 );

  value = 0.0;
  for ( i = 0; i <= n; i++ )
  {
    value = value + y[i] * bval[i];
  }

  delete [] bval;

  return value;
}
//****************************************************************************80

double bp_approx ( int n, double a, double b, double ydata[], double xval )

//****************************************************************************80
//
//  Purpose:
//
//    BP_APPROX evaluates the Bernstein polynomial for F(X) on [A,B].
//
//  Formula:
//
//    BERN(F)(X) = sum ( 0 <= I <= N ) F(X(I)) * B_BASE(I,X)
//
//    where
//
//      X(I) = ( ( N - I ) * A + I * B ) / N
//      B_BASE(I,X) is the value of the I-th Bernstein basis polynomial at X.
//
//  Discussion:
//
//    The Bernstein polynomial BERN(F) for F(X) is an approximant, not an
//    interpolant; in other words, its value is not guaranteed to equal
//    that of F at any particular point.  However, for a fixed interval
//    [A,B], if we let N increase, the Bernstein polynomial converges
//    uniformly to F everywhere in [A,B], provided only that F is continuous.
//    Even if F is not continuous, but is bounded, the polynomial converges
//    pointwise to F(X) at all points of continuity.  On the other hand,
//    the convergence is quite slow compared to other interpolation
//    and approximation schemes.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Kahaner, Cleve Moler, Steven Nash,
//    Numerical Methods and Software,
//    Prentice Hall, 1989,
//    ISBN: 0-13-627258-4,
//    LC: TA345.K34.
//
//  Parameters:
//
//    Input, int N, the degree of the Bernstein polynomial to be used.
//
//    Input, double A, B, the endpoints of the interval on which the
//    approximant is based.  A and B should not be equal.
//
//    Input, double YDATA[0:N], the data values at N+1 equally spaced points
//    in [A,B].  If N = 0, then the evaluation point should be 0.5 * ( A + B).
//    Otherwise, evaluation point I should be ( (N-I)*A + I*B ) / N ).
//
//    Input, double XVAL, the point at which the Bernstein polynomial
//    approximant is to be evaluated.  XVAL does not have to lie in the
//    interval [A,B].
//
//    Output, double BP_APPROX, the value of the Bernstein polynomial approximant
//    for F, based in [A,B], evaluated at XVAL.
//
{
  double *bvec;
  int i;
  double yval;
//
//  Evaluate the Bernstein basis polynomials at XVAL.
//
  bvec = bpab ( n, a, b, xval );
//
//  Now compute the sum of YDATA(I) * BVEC(I).
//
  yval = 0.0;

  for ( i = 0; i <= n; i++ )
  {
    yval = yval + ydata[i] * bvec[i];
  }
  delete [] bvec;

  return yval;
}
//****************************************************************************80

double *bp01 ( int n, double x )

//****************************************************************************80
//
//  Purpose:
//
//    BP01 evaluates the Bernstein basis polynomials for [0,1] at a point.
//
//  Discussion:
//
//    For any N greater than or equal to 0, there is a set of N+1 Bernstein
//    basis polynomials, each of degree N, which form a basis for
//    all polynomials of degree N on [0,1].
//
//  Formula:
//
//    BERN(N,I,X) = [N!/(I!*(N-I)!)] * (1-X)**(N-I) * X**I
//
//    N is the degree;
//
//    0 <= I <= N indicates which of the N+1 basis polynomials
//    of degree N to choose;
//
//    X is a point in [0,1] at which to evaluate the basis polynomial.
//
//  First values:
//
//    B(0,0,X) = 1
//
//    B(1,0,X) =      1-X
//    B(1,1,X) =                X
//
//    B(2,0,X) =     (1-X)**2
//    B(2,1,X) = 2 * (1-X)    * X
//    B(2,2,X) =                X**2
//
//    B(3,0,X) =     (1-X)**3
//    B(3,1,X) = 3 * (1-X)**2 * X
//    B(3,2,X) = 3 * (1-X)    * X**2
//    B(3,3,X) =                X**3
//
//    B(4,0,X) =     (1-X)**4
//    B(4,1,X) = 4 * (1-X)**3 * X
//    B(4,2,X) = 6 * (1-X)**2 * X**2
//    B(4,3,X) = 4 * (1-X)    * X**3
//    B(4,4,X) =                X**4
//
//  Special values:
//
//    B(N,I,1/2) = C(N,K) / 2**N
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Kahaner, Cleve Moler, Steven Nash,
//    Numerical Methods and Software,
//    Prentice Hall, 1989,
//    ISBN: 0-13-627258-4,
//    LC: TA345.K34.
//
//  Parameters:
//
//    Input, int N, the degree of the Bernstein basis polynomials.
//
//    Input, double X, the evaluation point.
//
//    Output, double BP01[0:N], the values of the N+1 Bernstein basis
//    polynomials at X.
//
{
  double *bern;
  int i;
  int j;

  bern = new double[n+1];

  if ( n == 0 )
  {
    bern[0] = 1.0;
  }
  else if ( 0 < n )
  {
    bern[0] = 1.0 - x;
    bern[1] = x;

    for ( i = 2; i <= n; i++ )
    {
      bern[i] = x * bern[i-1];
      for ( j = i-1; 1 <= j; j-- )
      {
        bern[j] = x * bern[j-1] + ( 1.0 - x ) * bern[j];
      }
      bern[0] = ( 1.0 - x ) * bern[0];
    }

  }

  return bern;
}
//****************************************************************************80

double *bpab ( int n, double a, double b, double x )

//****************************************************************************80
//
//  Purpose:
//
//    BPAB evaluates the Bernstein basis polynomials for [A,B] at a point.
//
//  Formula:
//
//    BERN(N,I,X) = [N!/(I!*(N-I)!)] * (B-X)**(N-I) * (X-A)**I / (B-A)**N
//
//  First values:
//
//    B(0,0,X) =   1
//
//    B(1,0,X) = (      B-X                ) / (B-A)
//    B(1,1,X) = (                 X-A     ) / (B-A)
//
//    B(2,0,X) = (     (B-X)**2            ) / (B-A)**2
//    B(2,1,X) = ( 2 * (B-X)    * (X-A)    ) / (B-A)**2
//    B(2,2,X) = (                (X-A)**2 ) / (B-A)**2
//
//    B(3,0,X) = (     (B-X)**3            ) / (B-A)**3
//    B(3,1,X) = ( 3 * (B-X)**2 * (X-A)    ) / (B-A)**3
//    B(3,2,X) = ( 3 * (B-X)    * (X-A)**2 ) / (B-A)**3
//    B(3,3,X) = (                (X-A)**3 ) / (B-A)**3
//
//    B(4,0,X) = (     (B-X)**4            ) / (B-A)**4
//    B(4,1,X) = ( 4 * (B-X)**3 * (X-A)    ) / (B-A)**4
//    B(4,2,X) = ( 6 * (B-X)**2 * (X-A)**2 ) / (B-A)**4
//    B(4,3,X) = ( 4 * (B-X)    * (X-A)**3 ) / (B-A)**4
//    B(4,4,X) = (                (X-A)**4 ) / (B-A)**4
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    David Kahaner, Cleve Moler, Steven Nash,
//    Numerical Methods and Software,
//    Prentice Hall, 1989,
//    ISBN: 0-13-627258-4,
//    LC: TA345.K34.
//
//  Parameters:
//
//    Input, integer N, the degree of the Bernstein basis polynomials.
//    For any N greater than or equal to 0, there is a set of N+1
//    Bernstein basis polynomials, each of degree N, which form a basis
//    for polynomials on [A,B].
//
//    Input, double A, B, the endpoints of the interval on which the
//    polynomials are to be based.  A and B should not be equal.
//
//    Input, double X, the point at which the polynomials are to be
//    evaluated.  X need not lie in the interval [A,B].
//
//    Output, double BERN[0:N], the values of the N+1 Bernstein basis
//    polynomials at X.
//
{
  double *bern;
  int i;
  int j;

  if ( b == a )
  {
    cout << "\n";
    cout << "BPAB - Fatal error!\n";
    cout << "  A = B = " << a << "\n";
    exit ( 1 );
  }

  bern = new double[n+1];

  if ( n == 0 )
  {
    bern[0] = 1.0;
  }
  else if ( 0 < n )
  {
    bern[0] = ( b - x ) / ( b - a );
    bern[1] = ( x - a ) / ( b - a );

    for ( i = 2; i <= n; i++ )
    {
      bern[i] = ( x - a ) * bern[i-1] / ( b - a );
      for ( j = i-1; 1 <= j; j-- )
      {
        bern[j] = ( ( b - x ) * bern[j] + ( x - a ) * bern[j-1] ) / ( b - a );
      }
      bern[0] = ( b - x ) * bern[0] / ( b - a );
    }
  }

  return bern;
}
//****************************************************************************80

int chfev ( double x1, double x2, double f1, double f2, double d1, double d2,
  int ne, double xe[], double fe[], int next[] )

//****************************************************************************80
//
//  Purpose:
//
//    CHFEV evaluates a cubic polynomial given in Hermite form.
//
//  Discussion:
//
//    This routine evaluates a cubic polynomial given in Hermite form at an
//    array of points.  While designed for use by SPLINE_PCHIP_VAL, it may
//    be useful directly as an evaluator for a piecewise cubic
//    Hermite function in applications, such as graphing, where
//    the interval is known in advance.
//
//    The cubic polynomial is determined by function values
//    F1, F2 and derivatives D1, D2 on the interval [X1,X2].
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 August 2005
//
//  Author:
//
//    Original FORTRAN77 version by Fred Fritsch, Lawrence Livermore National Laboratory.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Fred Fritsch, Ralph Carlson,
//    Monotone Piecewise Cubic Interpolation,
//    SIAM Journal on Numerical Analysis,
//    Volume 17, Number 2, April 1980, pages 238-246.
//
//    David Kahaner, Cleve Moler, Steven Nash,
//    Numerical Methods and Software,
//    Prentice Hall, 1989,
//    ISBN: 0-13-627258-4,
//    LC: TA345.K34.
//
//  Parameters:
//
//    Input, double X1, X2, the endpoints of the interval of
//    definition of the cubic.  X1 and X2 must be distinct.
//
//    Input, double F1, F2, the values of the function at X1 and
//    X2, respectively.
//
//    Input, double D1, D2, the derivative values at X1 and
//    X2, respectively.
//
//    Input, int NE, the number of evaluation points.
//
//    Input, double XE[NE], the points at which the function is to
//    be evaluated.  If any of the XE are outside the interval
//    [X1,X2], a warning error is returned in NEXT.
//
//    Output, double FE[NE], the value of the cubic function
//    at the points XE.
//
//    Output, int NEXT[2], indicates the number of extrapolation points:
//    NEXT[0] = number of evaluation points to the left of interval.
//    NEXT[1] = number of evaluation points to the right of interval.
//
//    Output, int CHFEV, error flag.
//    0, no errors.
//    -1, NE < 1.
//    -2, X1 == X2.
//
{
  double c2;
  double c3;
  double del1;
  double del2;
  double delta;
  double h;
  int i;
  int ierr;
  double x;
  double xma;
  double xmi;

  if ( ne < 1 )
  {
    ierr = -1;
    cout << "\n";
    cout << "CHFEV - Fatal error!\n";
    cout << "  Number of evaluation points is less than 1.\n";
    cout << "  NE = " << ne << "\n";
    return ierr;
  }

  h = x2 - x1;

  if ( h == 0.0 )
  {
    ierr = -2;
    cout << "\n";
    cout << "CHFEV - Fatal error!\n";
    cout << "  The interval [X1,X2] is of zero length.\n";
    return ierr;
  }
//
//  Initialize.
//
  ierr = 0;
  next[0] = 0;
  next[1] = 0;
  xmi = r8_min ( 0.0, h );
  xma = r8_max ( 0.0, h );
//
//  Compute cubic coefficients expanded about X1.
//
  delta = ( f2 - f1 ) / h;
  del1 = ( d1 - delta ) / h;
  del2 = ( d2 - delta ) / h;
  c2 = -( del1 + del1 + del2 );
  c3 = ( del1 + del2 ) / h;
//
//  Evaluation loop.
//
  for ( i = 0; i < ne; i++ )
  {
    x = xe[i] - x1;
    fe[i] = f1 + x * ( d1 + x * ( c2 + x * c3 ) );
//
//  Count the extrapolation points.
//
    if ( x < xmi )
    {
      next[0] = next[0] + 1;
    }

    if ( xma < x )
    {
      next[1] = next[1] + 1;
    }

  }

  return 0;
}
//****************************************************************************80

double *d3_mxv ( int n, double a[], double x[] )

//****************************************************************************80
//
//  Purpose:
//
//    D3_MXV multiplies a D3 matrix times a vector.
//
//  Discussion:
//
//    The D3 storage format is used for a tridiagonal matrix.
//    The superdiagonal is stored in entries (1,2:N), the diagonal in
//    entries (2,1:N), and the subdiagonal in (3,1:N-1).  Thus, the
//    original matrix is "collapsed" vertically into the array.
//
//  Example:
//
//    Here is how a D3 matrix of order 5 would be stored:
//
//       *  A12 A23 A34 A45
//      A11 A22 A33 A44 A55
//      A21 A32 A43 A54  *
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    15 November 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the order of the linear system.
//
//    Input, double A[3*N], the D3 matrix.
//
//    Input, double X[N], the vector to be multiplied by A.
//
//    Output, double D3_MXV[N], the product A * x.
//
{
  double *b;
  int i;

  b = new double[n];

  for ( i = 0; i < n; i++ )
  {
    b[i] =        a[1+i*3] * x[i];
  }
  for ( i = 0; i < n-1; i++ )
  {
    b[i] = b[i] + a[0+(i+1)*3] * x[i+1];
  }
  for ( i = 1; i < n; i++ )
  {
    b[i] = b[i] + a[2+(i-1)*3] * x[i-1];
  }

  return b;
}
//****************************************************************************80

double *d3_np_fs ( int n, double a[], double b[] )

//****************************************************************************80
//
//  Purpose:
//
//    D3_NP_FS factors and solves a D3 system.
//
//  Discussion:
//
//    The D3 storage format is used for a tridiagonal matrix.
//    The superdiagonal is stored in entries (1,2:N), the diagonal in
//    entries (2,1:N), and the subdiagonal in (3,1:N-1).  Thus, the
//    original matrix is "collapsed" vertically into the array.
//
//    This algorithm requires that each diagonal entry be nonzero.
//    It does not use pivoting, and so can fail on systems that
//    are actually nonsingular.
//
//  Example:
//
//    Here is how a D3 matrix of order 5 would be stored:
//
//       *  A12 A23 A34 A45
//      A11 A22 A33 A44 A55
//      A21 A32 A43 A54  *
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    15 November 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the order of the linear system.
//
//    Input/output, double A[3*N].
//    On input, the nonzero diagonals of the linear system.
//    On output, the data in these vectors has been overwritten
//    by factorization information.
//
//    Input, double B[N], the right hand side.
//
//    Output, double D3_NP_FS[N], the solution of the linear system.
//    This is NULL if there was an error because one of the diagonal
//    entries was zero.
//
{
  int i;
  double *x;
  double xmult;
//
//  Check.
//
  for ( i = 0; i < n; i++ )
  {
    if ( a[1+i*3] == 0.0 )
    {
      return NULL;
    }
  }
  x = new double[n];

  for ( i = 0; i < n; i++ )
  {
    x[i] = b[i];
  }

  for ( i = 1; i < n; i++ )
  {
    xmult = a[2+(i-1)*3] / a[1+(i-1)*3];
    a[1+i*3] = a[1+i*3] - xmult * a[0+i*3];
    x[i] = x[i] - xmult * x[i-1];
  }

  x[n-1] = x[n-1] / a[1+(n-1)*3];
  for ( i = n-2; 0 <= i; i-- )
  {
    x[i] = ( x[i] - a[0+(i+1)*3] * x[i+1] ) / a[1+i*3];
  }

  return x;
}
//****************************************************************************80

void d3_print ( int n, double a[], string title )

//****************************************************************************80
//
//  Purpose:
//
//    D3_PRINT prints a D3 matrix.
//
//  Discussion:
//
//    The D3 storage format is used for a tridiagonal matrix.
//    The superdiagonal is stored in entries (1,2:N), the diagonal in
//    entries (2,1:N), and the subdiagonal in (3,1:N-1).  Thus, the
//    original matrix is "collapsed" vertically into the array.
//
//  Example:
//
//    Here is how a D3 matrix of order 5 would be stored:
//
//       *  A12 A23 A34 A45
//      A11 A22 A33 A44 A55
//      A21 A32 A43 A54  *
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    20 September 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the order of the matrix.
//    N must be positive.
//
//    Input, double A[3*N], the D3 matrix.
//
//    Input, string TITLE, a title to print.
//
{
  if ( 0 < s_len_trim ( title ) )
  {
    cout << "\n";
    cout << title << "\n";
  }

  cout << "\n";

  d3_print_some ( n, a, 1, 1, n, n );

  return;
}
//****************************************************************************80

void d3_print_some ( int n, double a[], int ilo, int jlo, int ihi, int jhi )

//****************************************************************************80
//
//  Purpose:
//
//    D3_PRINT_SOME prints some of a D3 matrix.
//
//  Discussion:
//
//    The D3 storage format is used for a tridiagonal matrix.
//    The superdiagonal is stored in entries (1,2:N), the diagonal in
//    entries (2,1:N), and the subdiagonal in (3,1:N-1).  Thus, the
//    original matrix is "collapsed" vertically into the array.
//
//  Example:
//
//    Here is how a D3 matrix of order 5 would be stored:
//
//       *  A12 A23 A34 A45
//      A11 A22 A33 A44 A55
//      A21 A32 A43 A54  *
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 January 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the order of the matrix.
//    N must be positive.
//
//    Input, double A[3*N], the D3 matrix.
//
//    Input, int ILO, JLO, IHI, JHI, designate the first row and
//    column, and the last row and column, to be printed.
//
{
# define INCX 5

  int i;
  int i2hi;
  int i2lo;
  int inc;
  int j;
  int j2;
  int j2hi;
  int j2lo;
//
//  Print the columns of the matrix, in strips of 5.
//
  for ( j2lo = jlo; j2lo <= jhi; j2lo = j2lo + INCX )
  {
    j2hi = j2lo + INCX - 1;
    j2hi = i4_min ( j2hi, n );
    j2hi = i4_min ( j2hi, jhi );

    inc = j2hi + 1 - j2lo;

    cout << "\n";
    cout << "  Col: ";
    for ( j = j2lo; j <= j2hi; j++ )
    {
      j2 = j + 1 - j2lo;
      cout << setw(7) << j << "       ";
    }

    cout << "\n";
    cout << "  Row\n";
    cout << "  ---\n";
//
//  Determine the range of the rows in this strip.
//
    i2lo = i4_max ( ilo, 1 );
    i2lo = i4_max ( i2lo, j2lo - 1 );

    i2hi = i4_min ( ihi, n );
    i2hi = i4_min ( i2hi, j2hi + 1 );

    for ( i = i2lo; i <= i2hi; i++ )
    {
//
//  Print out (up to) 5 entries in row I, that lie in the current strip.
//
      cout << setw(6) << i << "  ";

      for ( j2 = 1; j2 <= inc; j2++ )
      {
        j = j2lo - 1 + j2;

        if ( 1 < i-j || 1 < j-i )
        {
          cout << "              ";
        }
        else if ( j == i+1 )
        {
          cout << setw(12) << a[0+(j-1)*3] << "  ";
        }
        else if ( j == i )
        {
          cout << setw(12) << a[1+(j-1)*3] << "  ";
        }
        else if ( j == i-1 )
        {
          cout << setw(12) << a[2+(j-1)*3] << "  ";
        }

      }
      cout << "\n";
    }
  }

  cout << "\n";

  return;
# undef INCX
}
//****************************************************************************80

double *d3_uniform ( int n, int *seed )

//****************************************************************************80
//
//  Purpose:
//
//    D3_UNIFORM randomizes a D3 matrix.
//
//  Discussion:
//
//    The D3 storage format is used for a tridiagonal matrix.
//    The superdiagonal is stored in entries (1,2:N), the diagonal in
//    entries (2,1:N), and the subdiagonal in (3,1:N-1).  Thus, the
//    original matrix is "collapsed" vertically into the array.
//
//  Example:
//
//    Here is how a D3 matrix of order 5 would be stored:
//
//       *  A12 A23 A34 A45
//      A11 A22 A33 A44 A55
//      A21 A32 A43 A54  *
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 January 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the order of the linear system.
//
//    Input/output, int *SEED, a seed for the random number generator.
//
//    Output, double D3_UNIFORM[3*N], the D3 matrix.
//
{
  double *a;
  int i;
  double *u;
  double *v;
  double *w;

  a = new double[3*n];

  u = r8vec_uniform ( n-1, 0.0, 1.0, seed );
  v = r8vec_uniform ( n,   0.0, 1.0, seed );
  w = r8vec_uniform ( n-1, 0.0, 1.0, seed );

  a[0+0*3] = 0.0;
  for ( i = 1; i < n; i++ )
  {
    a[0+i*3] = u[i-1];
  }
   for ( i = 0; i < n; i++ )
  {
    a[1+i*3] = v[i];
  }
  for ( i = 0; i < n-1; i++ )
  {
    a[2+i*3] = w[i];
  }
  a[2+(n-1)*3] = 0.0;

  delete [] u;
  delete [] v;
  delete [] w;

  return a;
}
//****************************************************************************80

void data_to_dif ( int ntab, double xtab[], double ytab[], double diftab[] )

//****************************************************************************80
//
//  Purpose:
//
//    DATA_TO_DIF sets up a divided difference table from raw data.
//
//  Discussion:
//
//    Space can be saved by using a single array for both the DIFTAB and
//    YTAB dummy parameters.  In that case, the difference table will
//    overwrite the Y data without interfering with the computation.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    04 September 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int NTAB, the number of pairs of points
//    (XTAB[I],YTAB[I]) which are to be used as data.
//
//    Input, double XTAB[NTAB], the X values at which data was taken.
//    These values must be distinct.
//
//    Input, double YTAB[NTAB], the corresponding Y values.
//
//    Output, double DIFTAB[NTAB], the divided difference coefficients
//    corresponding to the input (XTAB,YTAB).
//
{
  int i;
  int j;
//
//  Copy the data values into DIFTAB.
//
  for ( i = 0; i < ntab; i++ )
  {
    diftab[i] = ytab[i];
  }
//
//  Make sure the abscissas are distinct.
//
  for ( i = 0; i < ntab; i++ )
  {
    for ( j = i+1; j < ntab; j++ )
    {
      if ( xtab[i] - xtab[j] == 0.0 )
      {
        cout << "\n";
        cout << "DATA_TO_DIF - Fatal error!\n";
        cout << "  Two entries of XTAB are equal!\n";
        cout << "  XTAB[%d] = " << xtab[i] << "\n";
        cout << "  XTAB[%d] = " << xtab[j] << "\n";
        exit ( 1 );
      }
    }
  }
//
//  Compute the divided differences.
//
  for ( i = 1; i <= ntab-1; i++ )
  {
    for ( j = ntab-1; i <= j; j-- )
    {
      diftab[j] = ( diftab[j] - diftab[j-1] ) / ( xtab[j] - xtab[j-i] );
    }
  }

  return;
}
//****************************************************************************80

double dif_val ( int ntab, double xtab[], double diftab[], double xval )

//****************************************************************************80
//
//  Purpose:
//
//    DIF_VAL evaluates a divided difference polynomial at a point.
//
//  Discussion:
//
//    DATA_TO_DIF must be called first to set up the divided difference table.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 September 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, integer NTAB, the number of divided difference
//    coefficients in DIFTAB, and the number of points XTAB.
//
//    Input, double XTAB[NTAB], the X values upon which the
//    divided difference polynomial is based.
//
//    Input, double DIFTAB[NTAB], the divided difference table.
//
//    Input, double XVAL, a value of X at which the polynomial
//    is to be evaluated.
//
//    Output, double DIF_VAL, the value of the polynomial at XVAL.
//
{
  int i;
  double value;

  value = diftab[ntab-1];
  for ( i = 2; i <= ntab; i++ )
  {
    value = diftab[ntab-i] + ( xval - xtab[ntab-i] ) * value;
  }

  return value;
}
//****************************************************************************80

int i4_max ( int i1, int i2 )

//****************************************************************************80
//
//  Purpose:
//
//    I4_MAX returns the maximum of two I4's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 October 1998
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I1, I2, are two integers to be compared.
//
//    Output, int I4_MAX, the larger of I1 and I2.
//
//
{
  if ( i2 < i1 )
  {
    return i1;
  }
  else
  {
    return i2;
  }

}
//****************************************************************************80

int i4_min ( int i1, int i2 )

//****************************************************************************80
//
//  Purpose:
//
//    I4_MIN returns the smaller of two I4's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    13 October 1998
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int I1, I2, two integers to be compared.
//
//    Output, int I4_MIN, the smaller of I1 and I2.
//
//
{
  if ( i1 < i2 )
  {
    return i1;
  }
  else
  {
    return i2;
  }

}
//****************************************************************************80

void least_set ( int point_num, double x[], double f[], double w[],
  int nterms, double b[], double c[], double d[] )

//****************************************************************************80
//
//  Purpose:
//
//    LEAST_SET defines a least squares polynomial for given data.
//
//  Discussion:
//
//    This routine is based on ORTPOL by Conte and deBoor.
//
//    The polynomial may be evaluated at any point X by calling LEAST_VAL.
//
//    Thanks to Andrew Telford for pointing out a mistake in the form of
//    the check that there are enough unique data points, 25 June 2008.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 June 2008
//
//  Author:
//
//    Original FORTRAN77 version by Samuel Conte, Carl deBoor.
//    C++ version by John Burkardt
//
//  Reference:
//
//    Samuel Conte, Carl deBoor,
//    Elementary Numerical Analysis,
//    Second Edition,
//    McGraw Hill, 1972,
//    ISBN: 07-012446-4,
//    LC: QA297.C65.
//
//  Parameters:
//
//    Input, int POINT_NUM, the number of data values.
//
//    Input, double X[POINT_NUM], the abscissas of the data points.
//    At least NTERMS of the values in X must be distinct.
//
//    Input, double F[POINT_NUM], the data values at the points X(*).
//
//    Input, double W[POINT_NUM], the weights associated with
//    the data points.  Each entry of W should be positive.
//
//    Input, int NTERMS, the number of terms to use in the
//    approximating polynomial.  NTERMS must be at least 1.
//    The degree of the polynomial is NTERMS-1.
//
//    Output, double B[NTERMS], C[NTERMS], D[NTERMS], are quantities
//    defining the least squares polynomial for the input data,
//    which will be needed to evaluate the polynomial.
//
{
  int i;
  int j;
  int k;
  double p;
  double *pj;
  double *pjm1;
  double *s;
  double tol = 0.0;
  int unique_num;
//
//  Make sure at least NTERMS X values are unique.
//
  unique_num = r8vec_unique_count ( point_num, x, tol );

  if ( unique_num < nterms )
  {
    cout << "\n";
    cout << "LEAST_SET - Fatal error!\n";
    cout << "  The number of distinct X values must be\n";
    cout << "  at least NTERMS = " << nterms << "\n";
    cout << "  but the input data has only " << unique_num << "\n";
    cout << "  distinct entries.\n";
    return;
  }
//
//  Make sure all W entries are positive.
//
  for ( i = 0; i < point_num; i++ )
  {
    if ( w[i] <= 0.0 )
    {
      cout << "\n";
      cout << "LEAST_SET - Fatal error!\n";
      cout << "  All weights W must be positive,\n";
      cout << "  but weight " << i << "\n";
      cout << "  is " << w[i] << "\n";
      return;
    }
  }

  s = new double[nterms];
//
//  Start inner product summations at zero.
//
  r8vec_zero ( nterms, b );
  r8vec_zero ( nterms, c );
  r8vec_zero ( nterms, d );
  r8vec_zero ( nterms, s );
//
//  Set the values of P(-1,X) and P(0,X) at all data points.
//
  pjm1 = new double[point_num];
  pj = new double[point_num];

  r8vec_zero ( point_num, pjm1 );

  for ( i = 0; i < point_num; i++ )
  {
    pj[i] = 1.0;
  }
//
//  Now compute the value of P(J,X(I)) as
//
//    P(J,X(I)) = ( X(I) - B(J) ) * P(J-1,X(I)) - C(J) * P(J-2,X(I))
//
//  where
//
//    S(J) = < P(J,X), P(J,X) >
//    B(J) = < x*P(J,X), P(J,X) > / < P(J,X), P(J,X) >
//    C(J) = S(J) / S(J-1)
//
//  and the least squares coefficients are
//
//    D(J) = < F(X), P(J,X) > / < P(J,X), P(J,X) >
//
  for ( j = 1; j <= nterms; j++ )
  {
    for ( k = 0; k < point_num; k++ )
    {
      d[j-1] = d[j-1] + w[k] * f[k] * pj[k];
      b[j-1] = b[j-1] + w[k] * x[k] * pj[k] * pj[k];
      s[j-1] = s[j-1] + w[k] * pj[k] * pj[k];
    }

    d[j-1] = d[j-1] / s[j-1];

    if ( j == nterms )
    {
      c[j-1] = 0.0;
      return;
    }

    b[j-1] = b[j-1] / s[j-1];

    if ( j == 1 )
    {
      c[j-1] = 0.0;
    }
    else
    {
      c[j-1] = s[j-1] / s[j-2];
    }

    for ( i = 1; i <= point_num; i++ )
    {
      p = pj[i-1];
      pj[i-1] = ( x[i-1] - b[j-1] ) * pj[i-1] - c[j-1] * pjm1[i-1];
      pjm1[i-1] = p;
    }
  }

  delete [] pj;
  delete [] pjm1;

  return;
}
//****************************************************************************80

double least_val ( int nterms, double b[], double c[], double d[],
  double x )

//****************************************************************************80
//
//  Purpose:
//
//    LEAST_VAL evaluates a least squares polynomial defined by LEAST_SET.
//
//  Discussion:
//
//    The least squares polynomial is assumed to be defined as a sum
//
//      P(X) = SUM ( I = 1 to NTERMS ) D(I) * P(I-1,X)
//
//    where the orthogonal basis polynomials P(I,X) satisfy the following
//    three term recurrence:
//
//      P(-1,X) = 0
//      P(0,X) = 1
//      P(I,X) = ( X - B(I-1) ) * P(I-1,X) - C(I-1) * P(I-2,X)
//
//    Therefore, the least squares polynomial can be evaluated as follows:
//
//    If NTERMS is 1, then the value of P(X) is D(1) * P(0,X) = D(1).
//
//    Otherwise, P(X) is defined as the sum of NTERMS > 1 terms.  We can
//    reduce the number of terms by 1, because the polynomial P(NTERMS,X)
//    can be rewritten as a sum of polynomials;  Therefore, P(NTERMS,X)
//    can be eliminated from the sum, and its coefficient merged in with
//    those of other polynomials.  Repeat this process for P(NTERMS-1,X)
//    and so on until a single term remains.
//    P(NTERMS,X) of P(NTERMS-1,X)
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 October 2005
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Samuel Conte, Carl deBoor,
//    Elementary Numerical Analysis,
//    Second Edition,
//    McGraw Hill, 1972,
//    ISBN: 07-012446-4,
//    LC: QA297.C65.
//
//  Parameters:
//
//    Input, int NTERMS, the number of terms in the least squares
//    polynomial.  NTERMS must be at least 1.  The input value of NTERMS
//    may be reduced from the value given to R8POLY_LS_SET.  This will
//    evaluate the least squares polynomial of the lower degree specified.
//
//    Input, double B[NTERMS], C[NTERMS], D[NTERMS], the information
//    computed by R8POLY_LS_SET.
//
//    Input, double X, the point at which the least squares polynomial
//    is to be evaluated.
//
//    Output, double LEAST_VAL, the value of the least squares
//    polynomial at X.
//
{
  int i;
  double prev;
  double prev2;
  double px;

  px = d[nterms-1];
  prev = 0.0;

  for ( i = nterms-1; 1 <= i; i-- )
  {
    prev2 = prev;
    prev = px;

    if ( i == nterms-1 )
    {
      px = d[i-1] + ( x - b[i-1] ) * prev;
    }
    else
    {
      px = d[i-1] + ( x - b[i-1] ) * prev - c[i] * prev2;
    }
  }

  return px;
}
//****************************************************************************80

void least_val2 ( int nterms, double b[], double c[], double d[], double x,
  double *px, double *pxp )

//****************************************************************************80
//
//  Purpose:
//
//    LEAST_VAL2 evaluates a least squares polynomial defined by LEAST_SET.
//
//  Discussion:
//
//    This routine also computes the derivative of the polynomial.
//
//    The least squares polynomial is assumed to be defined as a sum
//
//      P(X) = SUM ( I = 1 to NTERMS ) D(I) * P(I-1,X)
//
//    where the orthogonal basis polynomials P(I,X) satisfy the following
//    three term recurrence:
//
//      P(-1,X) = 0
//      P(0,X) = 1
//      P(I,X) = ( X - B(I-1) ) * P(I-1,X) - C(I-1) * P(I-2,X)
//
//    Therefore, the least squares polynomial can be evaluated as follows:
//
//    If NTERMS is 1, then the value of P(X) is D(1) * P(0,X) = D(1).
//
//    Otherwise, P(X) is defined as the sum of NTERMS > 1 terms.  We can
//    reduce the number of terms by 1, because the polynomial P(NTERMS,X)
//    can be rewritten as a sum of polynomials;  Therefore, P(NTERMS,X)
//    can be eliminated from the sum, and its coefficient merged in with
//    those of other polynomials.  Repeat this process for P(NTERMS-1,X)
//    and so on until a single term remains.
//    P(NTERMS,X) of P(NTERMS-1,X)
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 October 2005
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int NTERMS, the number of terms in the least squares
//    polynomial.  NTERMS must be at least 1.  The value of NTERMS
//    may be reduced from the value given to R8POLY_LS_SET.
//    This will cause R8POLY_LS_VAL to evaluate the least squares polynomial
//    of the lower degree specified.
//
//    Input, double B[NTERMS], C[NTERMS], D[NTERMS], the information
//    computed by R8POLY_LS_SET.
//
//    Input, double X, the point at which the least squares polynomial
//    is to be evaluated.
//
//    Output, double *PX, *PXP, the value and derivative of the least
//    squares polynomial at X.
//
{
  int i;
  double pxm1;
  double pxm2;
  double pxpm1;
  double pxpm2;

  *px = d[nterms-1];
  *pxp = 0.0;
  pxm1 = 0.0;
  pxpm1 = 0.0;

  for ( i = nterms-1; 1 <= i; i-- )
  {
    pxm2 = pxm1;
    pxpm2 = pxpm1;
    pxm1 = *px;
    pxpm1 = *pxp;

    if ( i == nterms-1 )
    {
      *px = d[i-1] + ( x - b[i-1] ) * pxm1;
      *pxp = pxm1  + ( x - b[i-1] ) * pxpm1;
    }
    else
    {
      *px = d[i-1] + ( x - b[i-1] ) * pxm1  - c[i] * pxm2;
      *pxp = pxm1  + ( x - b[i-1] ) * pxpm1 - c[i] * pxpm2;
    }
  }
  return;
}
//****************************************************************************80

void least_set_old ( int ntab, double xtab[], double ytab[], int ndeg,
  double ptab[], double b[], double c[], double d[], double *eps, int *ierror )

//****************************************************************************80
//
//  Purpose:
//
//    LEAST_SET_OLD constructs the least squares polynomial approximation to data.
//
//  Discussion:
//
//    The least squares polynomial is not returned directly as a simple
//    polynomial.  Instead, it is represented in terms of a set of
//    orthogonal polynomials appopriate for the given data.  This makes
//    the computation more accurate, but means that the user can not
//    easily evaluate the computed polynomial.  Instead, the routine
//    LEAST_EVAL should be used to evaluate the least squares polynomial
//    at any point.  (However, the value of the least squares polynomial
//    at each of the data points is returned as part of this computation.)
//
//
//    A discrete unweighted inner product is used, so that
//
//      ( F(X), G(X) ) = sum ( 1 <= I <= NTAB ) F(XTAB(I)) * G(XTAB(I)).
//
//    The least squares polynomial is determined using a set of
//    orthogonal polynomials PHI.  These polynomials can be defined
//    recursively by:
//
//      PHI(0)(X) = 1
//      PHI(1)(X) = X - B(1)
//      PHI(I)(X) = ( X - B(I) ) * PHI(I-1)(X) - D(I) * PHI(I-2)(X)
//
//    The array B(1:NDEG) contains the values
//
//      B(I) = ( X*PHI(I-1), PHI(I-1) ) / ( PHI(I-1), PHI(I-1) )
//
//    The array D(2:NDEG) contains the values
//
//      D(I) = ( PHI(I-1), PHI(I-1) ) / ( PHI(I-2), PHI(I-2) )
//
//    Using this basis, the least squares polynomial can be represented as
//
//      P(X)(I) = sum ( 0 <= I <= NDEG ) C(I) * PHI(I)(X)
//
//    The array C(0:NDEG) contains the values
//
//      C(I) = ( YTAB(I), PHI(I) ) / ( PHI(I), PHI(I) )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    15 May 2004
//
//  Reference:
//
//    Gisela Engeln-Muellges, Frank Uhlig,
//    Numerical Algorithms with C,
//    Springer, 1996,
//    ISBN: 3-540-60530-4.
//
//  Parameters:
//
//    Input, int NTAB, the number of data points.
//
//    Input, double XTAB[NTAB], the X data.  The values in XTAB
//    should be distinct, and in increasing order.
//
//    Input, double YTAB[NTAB], the Y data values corresponding
//    to the X data in XTAB.
//
//    Input, int NDEG, the degree of the polynomial which the
//    program is to use.  NDEG must be at least 1, and less than or
//    equal to NTAB-1.
//
//    Output, double PTAB[NTAB], the value of the least squares polynomial
//    at the points XTAB(1:NTAB).
//
//    Output, double B[1:NDEG], C[0:NDEG], D[2:NDEG], arrays containing
//    data about the polynomial.
//
//    Output, double *EPS, the root-mean-square discrepancy of the
//    polynomial fit.
//
//    Output, int *IERROR, error flag.
//    zero, no error occurred;
//    nonzero, an error occurred, and the polynomial could not be computed.
//
{
  int B_OFFSET = -1;
  int D_OFFSET = -2;
  int i;
  int i0l1;
  int i1l1;
  int it;
  int k;
  int mdeg;
  double rn0;
  double rn1;
  double s;
  double sum2;
  double y_sum;
  double *ztab;

  *ierror = 0;
  ztab = new double[2*ntab];
//
//  Check NDEG.
//
  if ( ndeg < 1 )
  {
    *ierror = 1;
    cout << "\n";
    cout << "LEAST_SET_OLD - Fatal error!\n";
    cout << "  NDEG < 1.\n";
    exit ( 1 );
  }

  if ( ntab <= ndeg )
  {
    *ierror = 1;
    cout << "\n";
    cout << "LEAST_SET_OLD - Fatal error!\n";
    cout << "  NTAB <= NDEG.\n";
    exit ( 1 );
  }
//
//  Check that the abscissas are strictly increasing.
//
  for ( i = 1; i <= ntab-1; i++ )
  {
    if ( xtab[i] <= xtab[i-1] )
    {
      *ierror = 1;
      cout << "\n";
      cout << "LEAST_SET_OLD - Fatal error!\n";
      cout << "  XTAB must be strictly increasing, but\n";
      cout << "  XTAB(" << i-1 << ") = " << xtab[i-1] << "\n";
      cout << "  XTAB(" << i   << ") = " << xtab[i]   << "\n";
      exit ( 1 );
    }
  }

  i0l1 = 0;
  i1l1 = ntab;
//
//  The polynomial is of degree at least zero.
//
  y_sum = 0.0;
  for ( i = 0; i < ntab; i++ )
  {
    y_sum = y_sum + ytab[i];
  }

  rn0 = ntab;
  c[0] = y_sum / ( double ) ( ntab );

  for ( i = 0; i < ntab; i++ )
  {
    ptab[i] = y_sum / ( double ) ( ntab );
  }

  if ( ndeg == 0 )
  {
    *eps = 0.0;
    for ( i = 0; i < ntab; i++ )
    {
      *eps = *eps + pow ( ( y_sum / ( double ) ( ntab ) - ytab[i] ), 2 );
    }

    *eps = sqrt ( *eps / ( double ) ( ntab ) );
    delete [] ztab;
    return;
  }
//
//  The polynomial is of degree at least 1.
//
  ztab[0] = 0.0;
  for ( i = 0; i < ntab; i++ )
  {
    ztab[0] = ztab[0] + xtab[i];
  }

  b[1+B_OFFSET] = ztab[0] / ( double ) ( ntab );

  s = 0.0;
  sum2 = 0.0;
  for ( i = 0; i < ntab; i++ )
  {
    ztab[i1l1+i] = xtab[i] - b[1+B_OFFSET];
    s = s + ztab[i1l1+i] * ztab[i1l1+i];
    sum2 = sum2 + ztab[i1l1+i] * ( ytab[i] - ptab[i] );
  }

  rn1 = s;
  c[1] = sum2 / s;

  for ( i = 0; i < ntab; i++ )
  {
    ptab[i] = ptab[i] + c[1] * ztab[i1l1+i];
  }


  if ( ndeg == 1 )
  {
    *eps = 0.0;
    for ( i = 0; i < ntab; i++ )
    {
      *eps = *eps + pow ( ( ptab[i] - ytab[i] ), 2 );
    }

    *eps = sqrt ( *eps / ( double ) ( ntab ) );
    delete [] ztab;
    return;
  }

  for ( i = 0; i < ntab; i++ )
  {
    ztab[i] = 1.0;
  }

  mdeg = 2;
  k = 2;

  for ( ; ; )
  {
    d[k+D_OFFSET] = rn1 / rn0;

    sum2 = 0.0;
    for ( i = 0; i < ntab; i++ )
    {
      sum2 = sum2 + xtab[i] * ztab[i1l1+i] * ztab[i1l1+i];
    }

    b[k+B_OFFSET] = sum2 / rn1;

    s = 0.0;
    sum2 = 0.0;

    for ( i = 0; i < ntab; i++ )
    {
      ztab[i0l1+i] = ( xtab[i] - b[k+B_OFFSET] ) * ztab[i1l1+i]
        - d[k+D_OFFSET] * ztab[i0l1+i];
      s = s + ztab[i0l1+i] * ztab[i0l1+i];
      sum2 = sum2 + ztab[i0l1+i] * ( ytab[i] - ptab[i] );
    }

    rn0 = rn1;
    rn1 = s;

    c[k] = sum2 / rn1;

    it = i0l1;
    i0l1 = i1l1;
    i1l1 = it;

    for ( i = 0; i < ntab; i++ )
    {
      ptab[i] = ptab[i] + c[k] * ztab[i1l1+i];
    }

    if ( ndeg <= mdeg )
    {
      break;
    }

    mdeg = mdeg + 1;
    k = k + 1;

  }
//
//  Compute the RMS error.
//
  *eps = 0.0;
  for ( i = 0; i < ntab; i++ )
  {
    *eps = *eps + pow ( ( ptab[i] - ytab[i] ), 2 );
  }

  *eps = sqrt ( *eps / ( double ) ( ntab ) );
  delete [] ztab;

  return;
}
//****************************************************************************80

double least_val_old ( double x, int ndeg, double b[], double c[], double d[] )

//****************************************************************************80
//
//  Purpose:
//
//    LEAST_VAL_OLD evaluates a least squares polynomial defined by LEAST_SET_OLD.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    17 December 2004
//
//  Reference:
//
//    Gisela Engeln-Muellges, Frank Uhlig,
//    Numerical Algorithms with C,
//    Springer, 1996,
//    ISBN: 3-540-60530-4.
//
//  Parameters:
//
//    Input, double X, the point at which the polynomial is to be evaluated.
//
//    Input, int NDEG, the degree of the polynomial fit used.
//    This is the value of NDEG as returned from LEAST_SET_OLD.
//
//    Input, double B[1:NDEG], C[0:NDEG], D[2:NDEG], arrays defined by
//    LEAST_SET, and needed to evaluate the polynomial.
//
//    Output, double LEAST_VALPOLD, the value of the polynomial at X.
//
{
  int B_OFFSET = -1;
  int D_OFFSET = -2;
  int k=0;
  double sk = 0;
  double skp1 = 0;
  double skp2 = 0;
  double value = 0;

  if ( ndeg <= 0 )
  {
    value = c[0];
  }
  else if ( ndeg == 1 )
  {
    value = c[0] + c[1] * ( x - b[1+B_OFFSET] );
  }
  else
  {
    skp2 = c[ndeg];
    skp1 = c[ndeg-1] + ( x - b[ndeg+B_OFFSET] ) * skp2;

    for ( k = ndeg-2; 0 <= k; k-- )
    {
      sk = c[k] + ( x - b[k+1+B_OFFSET] ) * skp1 - d[k+2+D_OFFSET] * skp2;
      skp2 = skp1;
      skp1 = sk;
    }
    value = sk;
  }

  return value;
}
//****************************************************************************80

void parabola_val2 ( int ndim, int ndata, double tdata[], double ydata[],
  int left, double tval, double *yval )

//****************************************************************************80
//
//  Purpose:
//
//    PARABOLA_VAL2 evaluates a parabolic function through 3 points in a table.
//
//  Discussion:
//
//    This routine is a utility routine used by OVERHAUSER_SPLINE_VAL.
//    It constructs the parabolic interpolant through the data in
//    3 consecutive entries of a table and evaluates this interpolant
//    at a given abscissa value.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, integer NDIM, the dimension of a single data point.
//    NDIM must be at least 1.
//
//    Input, int NDATA, the number of data points.
//    NDATA must be at least 3.
//
//    Input, double TDATA[NDATA], the abscissas of the data points.  The
//    values in TDATA must be in strictly ascending order.
//
//    Input, double YDATA[NDIM*NDATA], the data points corresponding to
//    the abscissas.
//
//    Input, int LEFT, the location of the first of the three
//    consecutive data points through which the parabolic interpolant
//    must pass.  0 <= LEFT <= NDATA - 3.
//
//    Input, double TVAL, the value of T at which the parabolic interpolant
//    is to be evaluated.  Normally, TDATA[0] <= TVAL <= T[NDATA-1], and
//    the data will be interpolated.  For TVAL outside this range,
//    extrapolation will be used.
//
//    Output, double YVAL[NDIM], the value of the parabolic interpolant
//    at TVAL.
//
{
  double dif1;
  double dif2;
  int i;
  double t1;
  double t2;
  double t3;
  double y1;
  double y2;
  double y3;
//
//  Check.
//
  if ( left < 1 )
  {
    cout << "\n";
    cout << "PARABOLA_VAL2 - Fatal error!\n";
    cout << "  LEFT < 0.\n";
    exit ( 1 );
  }

  if ( ndata-2 < left )
  {
    cout << "\n";
    cout << "PARABOLA_VAL2 - Fatal error!\n";
    cout << " NDATA-2 < LEFT.\n";
    exit ( 1 );
  }

  if ( ndim < 1 )
  {
    cout << "\n";
    cout << "PARABOLA_VAL2 - Fatal error!\n";
    cout << " NDIM < 1.\n";
    exit ( 1 );
  }
//
//  Copy out the three abscissas.
//
  t1 = tdata[left-1];
  t2 = tdata[left];
  t3 = tdata[left+1];

  if ( t2 <= t1 || t3 <= t2 )
  {
    cout << "\n" ;
    cout << "PARABOLA_VAL2 - Fatal error!\n";
    cout << "  T2 <= T1 or T3 <= T2.\n";
    cout << "  T1 = " << t1 << "\n";
    cout << "  T2 = " << t2 << "\n";
    cout << "  T3 = " << t3 << "\n";
    exit ( 1 );
  }
//
//  Construct and evaluate a parabolic interpolant for the data.
//
  for ( i = 0; i < ndim; i++ )
  {
    y1 = ydata[i+(left-1)*ndim];
    y2 = ydata[i+(left  )*ndim];
    y3 = ydata[i+(left+1)*ndim];

    dif1 = ( y2 - y1 ) / ( t2 - t1 );
    dif2 =
      ( ( y3 - y1 ) / ( t3 - t1 )
      - ( y2 - y1 ) / ( t2 - t1 ) ) / ( t3 - t2 );

    yval[i] = y1 + ( tval - t1 ) * ( dif1 + ( tval - t2 ) * dif2 );
  }

  return;
}
//****************************************************************************80

double pchst ( double arg1, double arg2 )

//****************************************************************************80
//
//  Purpose:
//
//    PCHST: PCHIP sign-testing routine.
//
//  Discussion:
//
//    This routine essentially computes the sign of ARG1 * ARG2.
//
//    The object is to do this without multiplying ARG1 * ARG2, to avoid
//    possible over/underflow problems.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    12 August 2005
//
//  Author:
//
//    Original FORTRAN77 version by Fred Fritsch, Lawrence Livermore National Laboratory.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Fred Fritsch, Ralph Carlson,
//    Monotone Piecewise Cubic Interpolation,
//    SIAM Journal on Numerical Analysis,
//    Volume 17, Number 2, April 1980, pages 238-246.
//
//  Parameters:
//
//    Input, double ARG1, ARG2, two values to check.
//
//    Output, double PCHST,
//    -1.0, if ARG1 and ARG2 are of opposite sign.
//     0.0, if either argument is zero.
//    +1.0, if ARG1 and ARG2 are of the same sign.
//
{
  double value=0.0;

  if ( arg1 == 0.0 )
  {
    value = 0.0;
  }
  else if ( arg1 < 0.0 )
  {
    if ( arg2 < 0.0 )
    {
      value = 1.0;
    }
    else if ( arg2 == 0.0 )
    {
      value = 0.0;
    }
    else if ( 0.0 < arg2 )
    {
      value = -1.0;
    }
  }
  else if ( 0.0 < arg1 )
  {
    if ( arg2 < 0.0 )
    {
      value = -1.0;
    }
    else if ( arg2 == 0.0 )
    {
      value = 0.0;
    }
    else if ( 0.0 < arg2 )
    {
      value = 1.0;
    }
  }

  return value;
}
//****************************************************************************80

double r8_max ( double x, double y )

//****************************************************************************80
//
//  Purpose:
//
//    R8_MAX returns the maximum of two R8's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    10 January 2002
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, Y, the quantities to compare.
//
//    Output, double R8_MAX, the maximum of X and Y.
//
{
  if ( y < x )
  {
    return x;
  }
  else
  {
    return y;
  }
}
//****************************************************************************80

double r8_min ( double x, double y )

//****************************************************************************80
//
//  Purpose:
//
//    R8_MIN returns the minimum of two R8's.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    09 May 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double X, Y, the quantities to compare.
//
//    Output, double R8_MIN, the minimum of X and Y.
//
{
  if ( y < x )
  {
    return y;
  }
  else
  {
    return x;
  }
}
//****************************************************************************80

double r8_uniform_01 ( int *seed )

//****************************************************************************80
//
//  Purpose:
//
//    R8_UNIFORM_01 is a portable pseudorandom number generator.
//
//  Discussion:
//
//    This routine implements the recursion
//
//      seed = 16807 * seed mod ( 2**31 - 1 )
//      unif = seed / ( 2**31 - 1 )
//
//    The integer arithmetic never requires more than 32 bits,
//    including a sign bit.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 August 2004
//
//  Reference:
//
//    Paul Bratley, Bennett Fox, Linus Schrage,
//    A Guide to Simulation,
//    Springer Verlag, pages 201-202, 1983.
//
//    Bennett Fox,
//    Algorithm 647:
//    Implementation and Relative Efficiency of Quasirandom
//    Sequence Generators,
//    ACM Transactions on Mathematical Software,
//    Volume 12, Number 4, pages 362-376, 1986.
//
//  Parameters:
//
//    Input/output, int *SEED, the "seed" value.  Normally, this
//    value should not be 0.  On output, SEED has been updated.
//
//    Output, double R8_UNIFORM_01, a new pseudorandom variate, strictly between
//    0 and 1.
//
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }
//
//  Although SEED can be represented exactly as a 32 bit integer,
//  it generally cannot be represented exactly as a 32 bit real number!
//
  r = ( double ) ( *seed ) * 4.656612875E-10;

  return r;
}
//****************************************************************************80

void r8vec_bracket ( int n, double x[], double xval, int *left,
  int *right )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_BRACKET searches a sorted array for successive brackets of a value.
//
//  Discussion:
//
//    If the values in the vector are thought of as defining intervals
//    on the real line, then this routine searches for the interval
//    nearest to or containing the given value.
//
//    It is always true that RIGHT = LEFT+1.
//
//    If XVAL < X[0], then LEFT = 1, RIGHT = 2, and
//      XVAL   < X[0] < X[1];
//    If X(1) <= XVAL < X[N-1], then
//      X[LEFT-1] <= XVAL < X[RIGHT-1];
//    If X[N-1] <= XVAL, then LEFT = N-1, RIGHT = N, and
//      X[LEFT-1] <= X[RIGHT-1] <= XVAL.
//
//    For consistency, this routine computes indices RIGHT and LEFT
//    that are 1-based, although it would be more natural in C and
//    C++ to use 0-based values.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, length of input array.
//
//    Input, double X[N], an array that has been sorted into ascending order.
//
//    Input, double XVAL, a value to be bracketed.
//
//    Output, int *LEFT, *RIGHT, the results of the search.
//
{
  int i;

  for ( i = 2; i <= n - 1; i++ )
  {
    if ( xval < x[i-1] )
    {
      *left = i - 1;
      *right = i;
      return;
    }

   }

  *left = n - 1;
  *right = n;

  return;
}
//****************************************************************************80

void r8vec_bracket3 ( int n, double t[], double tval, int *left )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_BRACKET3 finds the interval containing or nearest a given value.
//
//  Discussion:
//
//    The routine always returns the index LEFT of the sorted array
//    T with the property that either
//    *  T is contained in the interval [ T[LEFT], T[LEFT+1] ], or
//    *  T < T[LEFT] = T[0], or
//    *  T > T[LEFT+1] = T[N-1].
//
//    The routine is useful for interpolation problems, where
//    the abscissa must be located within an interval of data
//    abscissas for interpolation, or the "nearest" interval
//    to the (extreme) abscissa must be found so that extrapolation
//    can be carried out.
//
//    For consistency with other versions of this routine, the
//    value of LEFT is assumed to be a 1-based index.  This is
//    contrary to the typical C and C++ convention of 0-based indices.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, length of the input array.
//
//    Input, double T[N], an array that has been sorted into ascending order.
//
//    Input, double TVAL, a value to be bracketed by entries of T.
//
//    Input/output, int *LEFT.
//
//    On input, if 1 <= LEFT <= N-1, LEFT is taken as a suggestion for the
//    interval [ T[LEFT-1] T[LEFT] ] in which TVAL lies.  This interval
//    is searched first, followed by the appropriate interval to the left
//    or right.  After that, a binary search is used.
//
//    On output, LEFT is set so that the interval [ T[LEFT-1], T[LEFT] ]
//    is the closest to TVAL; it either contains TVAL, or else TVAL
//    lies outside the interval [ T[0], T[N-1] ].
//
{
  int high;
  int low;
  int mid;
//
//  Check the input data.
//
  if ( n < 2 )
  {
    cout << "\n";
    cout << "R8VEC_BRACKET3 - Fatal error!\n";
    cout << "  N must be at least 2.\n";
    exit ( 1 );
  }
//
//  If *LEFT is not between 1 and N-1, set it to the middle value.
//
  if ( *left < 1 || n - 1 < *left )
  {
    *left = ( n + 1 ) / 2;
  }

//
//  CASE 1: TVAL < T[*LEFT]:
//  Search for TVAL in (T[I],T[I+1]), for I = 1 to *LEFT-1.
//
  if ( tval < t[*left] )
  {

    if ( *left == 1 )
    {
      return;
    }
    else if ( *left == 2 )
    {
      *left = 1;
      return;
    }
    else if ( t[*left-2] <= tval )
    {
      *left = *left - 1;
      return;
    }
    else if ( tval <= t[1] )
    {
      *left = 1;
      return;
    }
//
//  ...Binary search for TVAL in (T[I-1],T[I]), for I = 2 to *LEFT-2.
//
    low = 2;
    high = *left - 2;

    for (;;)
    {

      if ( low == high )
      {
        *left = low;
        return;
      }

      mid = ( low + high + 1 ) / 2;

      if ( t[mid-1] <= tval )
      {
        low = mid;
      }
      else
      {
        high = mid - 1;
      }

    }
  }
//
//  CASE 2: T[*LEFT] < TVAL:
//  Search for TVAL in (T[I-1],T[I]) for intervals I = *LEFT+1 to N-1.
//
  else if ( t[*left] < tval )
  {

    if ( *left == n - 1 )
    {
      return;
    }
    else if ( *left == n - 2 )
    {
      *left = *left + 1;
      return;
    }
    else if ( tval <= t[*left+1] )
    {
      *left = *left + 1;
      return;
    }
    else if ( t[n-2] <= tval )
    {
      *left = n - 1;
      return;
    }
//
//  ...Binary search for TVAL in (T[I-1],T[I]) for intervals I = *LEFT+2 to N-2.
//
    low = *left + 2;
    high = n - 2;

    for ( ; ; )
    {

      if ( low == high )
      {
        *left = low;
        return;
      }

      mid = ( low + high + 1 ) / 2;

      if ( t[mid-1] <= tval )
      {
        low = mid;
      }
      else
      {
        high = mid - 1;
      }
    }
  }
//
//  CASE 3: T[*LEFT-1] <= TVAL <= T[*LEFT]:
//  T is just where the user said it might be.
//
  else
  {
  }

  return;
}
//****************************************************************************80

double *r8vec_even ( int n, double alo, double ahi )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_EVEN returns N real values, evenly spaced between ALO and AHI.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    17 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of values.
//
//    Input, double ALO, AHI, the low and high values.
//
//    Output, double R8VEC_EVEN[N], N evenly spaced values.
//    Normally, A(1) = ALO and A(N) = AHI.
//    However, if N = 1, then A(1) = 0.5*(ALO+AHI).
//
{
  double *a;
  int i;

  a = new double[n];

  if ( n == 1 )
  {
    a[0] = 0.5 * ( alo + ahi );
  }
  else
  {
    for ( i = 1; i <= n; i++ )
    {
      a[i-1] = ( ( double ) ( n - i     ) * alo
               + ( double ) (     i - 1 ) * ahi )
               / ( double ) ( n     - 1 );
    }
  }

  return a;
}
//****************************************************************************80

double *r8vec_indicator ( int n )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_INDICATOR sets an R8VEC to the indicator vector.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 September 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of elements of A.
//
//    Output, double R8VEC_INDICATOR[N], the array to be initialized.
//
{
  int i;
  double *a;

  a = new double[n];

  for ( i = 0; i < n; i++ )
  {
    a[i] = ( double ) ( i + 1 );
  }

  return a;
}
//****************************************************************************80

void r8vec_order_type ( int n, double x[], int *order )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_ORDER_TYPE determines if an R8VEC is (non)strictly ascending/descending.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 September 2000
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of entries of the array.
//
//    Input, double X[N], the array to be checked.
//
//    Output, int *ORDER, order indicator:
//    -1, no discernable order;
//    0, all entries are equal;
//    1, ascending order;
//    2, strictly ascending order;
//    3, descending order;
//    4, strictly descending order.
//
{
  int i;
//
//  Search for the first value not equal to X(0).
//
  i = 0;

  for (;;)
  {

    i = i + 1;
    if ( n-1 < i )
    {
      *order = 0;
      return;
    }

    if ( x[0] < x[i] )
    {
      if ( i == 1 )
      {
        *order = 2;
        break;
      }
      else
      {
        *order = 1;
        break;
      }
    }
    else if ( x[i] < x[0] )
    {
      if ( i == 1 )
      {
        *order = 4;
        break;
      }
      else
      {
        *order = 3;
        break;
      }
    }
  }
//
//  Now we have a "direction".  Examine subsequent entries.
//
  for (;;)
  {
    i = i + 1;
    if ( n - 1 < i )
    {
      return;
    }

    if ( *order == 1 )
    {
      if ( x[i] < x[i-1] )
      {
        *order = -1;
        return;
      }
    }
    else if ( *order == 2 )
    {
      if ( x[i] < x[i-1] )
      {
        *order = -1;
        return;
      }
      else if ( x[i] == x[i-1] )
      {
        *order = 1;
      }
    }
    else if ( *order == 3 )
    {
      if ( x[i-1] < x[i] )
      {
        *order = -1;
        return;
      }
    }
    else if ( *order == 4 )
    {
      if ( x[i-1] < x[i] )
      {
        *order = -1;
        return;
      }
      else if ( x[i] == x[i-1] )
      {
        *order = 3;
      }
    }
  }
}
//****************************************************************************80

void r8vec_print ( int n, double a[], string title )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_PRINT prints an R8VEC.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 November 2003
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of components of the vector.
//
//    Input, double A[N], the vector to be printed.
//
//    Input, string TITLE, a title to be printed first.
//    TITLE may be blank.
//
{
  int i;

  if ( s_len_trim ( title ) != 0 )
  {
    cout << "\n";
    cout << title << "\n";
  }

  cout << "\n";
  for ( i = 0; i <= n-1; i++ )
  {
    cout << setw(6)  << i + 1 << "  "
         << setw(14) << a[i]  << "\n";
  }

  return;
}
//****************************************************************************80

void r8vec_sort_bubble_a ( int n, double a[] )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_SORT_BUBBLE_A ascending sorts an R8VEC using bubble sort.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    09 April 1999
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, length of input array.
//
//    Input/output, double A[N].
//    On input, an unsorted array of floats.
//    On output, A has been sorted.
//
{
  int i;
  int j;
  double temp;

  for ( i = 0; i < n-1; i++ )
  {
    for ( j = i+1; j < n; j++ )
    {
      if ( a[j] < a[i] )
      {
        temp = a[i];
        a[i] = a[j];
        a[j] = temp;
      }
    }
  }
}
//****************************************************************************80

double *r8vec_uniform ( int n, double b, double c, int *seed )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_UNIFORM returns a scaled pseudorandom R8VEC.
//
//  Discussion:
//
//    This routine implements the recursion
//
//      seed = 16807 * seed mod ( 2**31 - 1 )
//      unif = seed / ( 2**31 - 1 )
//
//    The integer arithmetic never requires more than 32 bits,
//    including a sign bit.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    30 January 2005
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Paul Bratley, Bennett Fox, Linus Schrage,
//    A Guide to Simulation,
//    Springer Verlag, pages 201-202, 1983.
//
//    Bennett Fox,
//    Algorithm 647:
//    Implementation and Relative Efficiency of Quasirandom
//    Sequence Generators,
//    ACM Transactions on Mathematical Software,
//    Volume 12, Number 4, pages 362-376, 1986.
//
//    Peter Lewis, Allen Goodman, James Miller,
//    A Pseudo-Random Number Generator for the System/360,
//    IBM Systems Journal,
//    Volume 8, pages 136-143, 1969.
//
//  Parameters:
//
//    Input, int N, the number of entries in the vector.
//
//    Input, double B, C, the lower and upper limits of the pseudorandom values.
//
//    Input/output, int *SEED, a seed for the random number generator.
//
//    Output, double R8VEC_UNIFORM_01[N], the vector of pseudorandom values.
//
{
  int i;
  int k;
  double *r;

  r = new double[n];

  for ( i = 0; i < n; i++ )
  {
    k = *seed / 127773;

    *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

    if ( *seed < 0 )
    {
      *seed = *seed + 2147483647;
    }

    r[i] = b + ( c - b ) * ( double ) ( *seed ) * 4.656612875E-10;
  }

  return r;
}
//****************************************************************************80

int r8vec_unique_count ( int n, double a[], double tol )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_UNIQUE_COUNT counts the unique elements in an unsorted real array.
//
//  Discussion:
//
//    Because the array is unsorted, this algorithm is O(N^2).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    29 April 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of elements of A.
//
//    Input, double A[N], the array to examine, which does NOT have to
//    be sorted.
//
//    Input, double TOL, a tolerance for checking equality.
//
//    Output, int R8VEC_UNIQUE_COUNT, the number of unique elements of A.
//
{
  int i;
  int j;
  int unique_num;

  unique_num = 0;

  for ( i = 0; i < n; i++ )
  {
    unique_num = unique_num + 1;

    for ( j = 0; j < i; j++ )
    {
      if ( fabs ( a[i] - a[j] ) <= tol )
      {
        unique_num = unique_num - 1;
        break;
      }
    }
  }

  return unique_num;
}
//****************************************************************************80

void r8vec_zero ( int n, double a[] )

//****************************************************************************80
//
//  Purpose:
//
//    R8VEC_ZERO zeroes an R8VEC.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    03 July 2005
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of entries in the vector.
//
//    Output, double A[N], a vector of zeroes.
//
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    a[i] = 0.0;
  }
  return;
}
//****************************************************************************80

int s_len_trim ( string s )

//****************************************************************************80
//
//  Purpose:
//
//    S_LEN_TRIM returns the length of a string to the last nonblank.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    05 July 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, string S, a string.
//
//    Output, int S_LEN_TRIM, the length of the string to the last nonblank.
//    If S_LEN_TRIM is 0, then the string is entirely blank.
//
{
  int n;

  n = (int)s.length ( );

  while ( 0 < n )
  {
    if ( s[n-1] != ' ' )
    {
      return n;
    }
    n = n - 1;
  }

  return n;
}
//****************************************************************************80

double spline_b_val ( int ndata, double tdata[], double ydata[], double tval )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_B_VAL evaluates a cubic B spline approximant.
//
//  Discussion:
//
//    The cubic B spline will approximate the data, but is not
//    designed to interpolate it.
//
//    In effect, two "phantom" data values are appended to the data,
//    so that the spline will interpolate the first and last data values.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Carl deBoor,
//    A Practical Guide to Splines,
//    Springer, 2001,
//    ISBN: 0387953663.
//
//  Parameters:
//
//    Input, int NDATA, the number of data values.
//
//    Input, double TDATA[NDATA], the abscissas of the data.
//
//    Input, double YDATA[NDATA], the data values.
//
//    Input, double TVAL, a point at which the spline is to be evaluated.
//
//    Output, double SPLINE_B_VAL, the value of the function at TVAL.
//
{
  double bval;
  int left;
  int right;
  double u;
  double yval;
//
//  Find the nearest interval [ TDATA(LEFT), TDATA(RIGHT) ] to TVAL.
//
  r8vec_bracket ( ndata, tdata, tval, &left, &right );
//
//  Evaluate the 5 nonzero B spline basis functions in the interval,
//  weighted by their corresponding data values.
//
  u = ( tval - tdata[left-1] ) / ( tdata[right-1] - tdata[left-1] );
  yval = 0.0;
//
//  B function associated with node LEFT - 1, (or "phantom node"),
//  evaluated in its 4th interval.
//
  bval = ( ( (     - 1.0
               * u + 3.0 )
               * u - 3.0 )
               * u + 1.0 ) / 6.0;

  if ( 0 < left-1 )
  {
    yval = yval + ydata[left-2] * bval;
  }
  else
  {
    yval = yval + ( 2.0 * ydata[0] - ydata[1] ) * bval;
  }
//
//  B function associated with node LEFT,
//  evaluated in its third interval.
//
  bval = ( ( (       3.0
               * u - 6.0 )
               * u + 0.0 )
               * u + 4.0 ) / 6.0;

  yval = yval + ydata[left-1] * bval;
//
//  B function associated with node RIGHT,
//  evaluated in its second interval.
//
  bval = ( ( (     - 3.0
               * u + 3.0 )
               * u + 3.0 )
               * u + 1.0 ) / 6.0;

  yval = yval + ydata[right-1] * bval;
//
//  B function associated with node RIGHT+1, (or "phantom node"),
//  evaluated in its first interval.
//
  bval = pow ( u, 3 ) / 6.0;

  if ( right+1 <= ndata )
  {
    yval = yval + ydata[right] * bval;
  }
  else
  {
    yval = yval + ( 2.0 * ydata[ndata-1] - ydata[ndata-2] ) * bval;
  }

  return yval;
}
//****************************************************************************80

double spline_beta_val ( double beta1, double beta2, int ndata, double tdata[],
  double ydata[], double tval )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_BETA_VAL evaluates a cubic beta spline approximant.
//
//  Discussion:
//
//    The cubic beta spline will approximate the data, but is not
//    designed to interpolate it.
//
//    If BETA1 = 1 and BETA2 = 0, the cubic beta spline will be the
//    same as the cubic B spline approximant.
//
//    With BETA1 = 1 and BETA2 large, the beta spline becomes more like
//    a linear spline.
//
//    In effect, two "phantom" data values are appended to the data,
//    so that the spline will interpolate the first and last data values.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, double BETA1, the skew or bias parameter.
//    BETA1 = 1 for no skew or bias.
//
//    Input, double BETA2, the tension parameter.
//    BETA2 = 0 for no tension.
//
//    Input, int NDATA, the number of data values.
//
//    Input, double TDATA[NDATA], the abscissas of the data.
//
//    Input, double YDATA[NDATA], the data values.
//
//    Input, double TVAL, a point at which the spline is to be evaluated.
//
//    Output, double SPLINE_BETA_VAL, the value of the function at TVAL.
//
{
  double a;
  double b;
  double bval;
  double c;
  double d;
  double delta;
  int left;
  int right;
  double u;
  double yval;
//
//  Find the nearest interval [ TDATA(LEFT), TDATA(RIGHT) ] to TVAL.
//
  r8vec_bracket ( ndata, tdata, tval, &left, &right );
//
//  Evaluate the 5 nonzero beta spline basis functions in the interval,
//  weighted by their corresponding data values.
//
  u = ( tval - tdata[left-1] ) / ( tdata[right-1] - tdata[left-1] );

  delta = ( ( 2.0
    * beta1 + 4.0 )
    * beta1 + 4.0 )
    * beta1 + 2.0 + beta2;

  yval = 0.0;
//
//  Beta function associated with node LEFT - 1, (or "phantom node"),
//  evaluated in its 4th interval.
//
  bval = 2.0 * pow ( ( beta1 * ( 1.0 - u ) ), 3 )  / delta;

  if ( 0 < left-1 )
  {
    yval = yval + ydata[left-2] * bval;
  }
  else
  {
    yval = yval + ( 2.0 * ydata[0] - ydata[1] ) * bval;
  }
//
//  Beta function associated with node LEFT,
//  evaluated in its third interval.
//
  a = beta2 + ( 4.0 + 4.0 * beta1 ) * beta1;

  b = - 6.0 * beta1 * ( 1.0 - beta1 ) * ( 1.0 + beta1 );

  c = ( (     - 6.0
      * beta1 - 6.0 )
      * beta1 + 0.0 )
      * beta1 - 3.0 * beta2;

  d = ( (     + 2.0
      * beta1 + 2.0 )
      * beta1 + 2.0 )
      * beta1 + 2.0 * beta2;

  bval = ( a + u * ( b + u * ( c + u * d ) ) ) / delta;

  yval = yval + ydata[left-1] * bval;
//
//  Beta function associated with node RIGHT,
//  evaluated in its second interval.
//
  a = 2.0;

  b = 6.0 * beta1;

  c = 3.0 * beta2 + 6.0 * beta1 * beta1;

  d = - 2.0 * ( 1.0 + beta2 + beta1 + beta1 * beta1 );

  bval = ( a + u * ( b + u * ( c + u * d ) ) ) / delta;

  yval = yval + ydata[right-1] * bval;
//
//  Beta function associated with node RIGHT+1, (or "phantom node"),
//  evaluated in its first interval.
//
  bval = 2.0 * pow ( u, 3 ) / delta;

  if ( right+1 <= ndata )
  {
    yval = yval + ydata[right] * bval;
  }
  else
  {
    yval = yval + ( 2.0 * ydata[ndata-1] - ydata[ndata-2] ) * bval;
  }

  return yval;
}
//****************************************************************************80

double spline_constant_val ( int ndata, double tdata[], double ydata[],
  double tval )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_CONSTANT_VAL evaluates a piecewise constant spline at a point.
//
//  Discussion:
//
//    NDATA-1 points TDATA define NDATA intervals, with the first
//    and last being semi-infinite.
//
//    The value of the spline anywhere in interval I is YDATA(I).
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    10 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int NDATA, the number of data points defining the spline.
//
//    Input, double TDATA[NDATA-1], the breakpoints.  The values of TDATA should
//    be distinct and increasing.
//
//    Input, double YDATA[NDATA], the values of the spline in the intervals
//    defined by the breakpoints.
//
//    Input, double TVAL, the point at which the spline is to be evaluated.
//
//    Output, double *SPLINE_CONSTANT_VAL, the value of the spline at TVAL.
//
{
  int i;

  for ( i = 0; i < ndata-1; i++ )
  {
    if ( tval <= tdata[i] )
    {
      return ydata[i];
    }
  }

  return ydata[ndata-1];
}
//****************************************************************************80

double *spline_cubic_set ( int n, double t[], double y[], int ibcbeg,
  double ybcbeg, int ibcend, double ybcend )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_CUBIC_SET computes the second derivatives of a piecewise cubic spline.
//
//  Discussion:
//
//    For data interpolation, the user must call SPLINE_SET to determine
//    the second derivative data, passing in the data to be interpolated,
//    and the desired boundary conditions.
//
//    The data to be interpolated, plus the SPLINE_SET output, defines
//    the spline.  The user may then call SPLINE_VAL to evaluate the
//    spline at any point.
//
//    The cubic spline is a piecewise cubic polynomial.  The intervals
//    are determined by the "knots" or abscissas of the data to be
//    interpolated.  The cubic spline has continous first and second
//    derivatives over the entire interval of interpolation.
//
//    For any point T in the interval T(IVAL), T(IVAL+1), the form of
//    the spline is
//
//      SPL(T) = A(IVAL)
//             + B(IVAL) * ( T - T(IVAL) )
//             + C(IVAL) * ( T - T(IVAL) )**2
//             + D(IVAL) * ( T - T(IVAL) )**3
//
//    If we assume that we know the values Y(*) and YPP(*), which represent
//    the values and second derivatives of the spline at each knot, then
//    the coefficients can be computed as:
//
//      A(IVAL) = Y(IVAL)
//      B(IVAL) = ( Y(IVAL+1) - Y(IVAL) ) / ( T(IVAL+1) - T(IVAL) )
//        - ( YPP(IVAL+1) + 2 * YPP(IVAL) ) * ( T(IVAL+1) - T(IVAL) ) / 6
//      C(IVAL) = YPP(IVAL) / 2
//      D(IVAL) = ( YPP(IVAL+1) - YPP(IVAL) ) / ( 6 * ( T(IVAL+1) - T(IVAL) ) )
//
//    Since the first derivative of the spline is
//
//      SPL'(T) =     B(IVAL)
//              + 2 * C(IVAL) * ( T - T(IVAL) )
//              + 3 * D(IVAL) * ( T - T(IVAL) )**2,
//
//    the requirement that the first derivative be continuous at interior
//    knot I results in a total of N-2 equations, of the form:
//
//      B(IVAL-1) + 2 C(IVAL-1) * (T(IVAL)-T(IVAL-1))
//      + 3 * D(IVAL-1) * (T(IVAL) - T(IVAL-1))**2 = B(IVAL)
//
//    or, setting H(IVAL) = T(IVAL+1) - T(IVAL)
//
//      ( Y(IVAL) - Y(IVAL-1) ) / H(IVAL-1)
//      - ( YPP(IVAL) + 2 * YPP(IVAL-1) ) * H(IVAL-1) / 6
//      + YPP(IVAL-1) * H(IVAL-1)
//      + ( YPP(IVAL) - YPP(IVAL-1) ) * H(IVAL-1) / 2
//      =
//      ( Y(IVAL+1) - Y(IVAL) ) / H(IVAL)
//      - ( YPP(IVAL+1) + 2 * YPP(IVAL) ) * H(IVAL) / 6
//
//    or
//
//      YPP(IVAL-1) * H(IVAL-1) + 2 * YPP(IVAL) * ( H(IVAL-1) + H(IVAL) )
//      + YPP(IVAL) * H(IVAL)
//      =
//      6 * ( Y(IVAL+1) - Y(IVAL) ) / H(IVAL)
//      - 6 * ( Y(IVAL) - Y(IVAL-1) ) / H(IVAL-1)
//
//    Boundary conditions must be applied at the first and last knots.
//    The resulting tridiagonal system can be solved for the YPP values.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    06 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of data points.  N must be at least 2.
//    In the special case where N = 2 and IBCBEG = IBCEND = 0, the
//    spline will actually be linear.
//
//    Input, double T[N], the knot values, that is, the points were data is
//    specified.  The knot values should be distinct, and increasing.
//
//    Input, double Y[N], the data values to be interpolated.
//
//    Input, int IBCBEG, left boundary condition flag:
//      0: the cubic spline should be a quadratic over the first interval;
//      1: the first derivative at the left endpoint should be YBCBEG;
//      2: the second derivative at the left endpoint should be YBCBEG.
//
//    Input, double YBCBEG, the values to be used in the boundary
//    conditions if IBCBEG is equal to 1 or 2.
//
//    Input, int IBCEND, right boundary condition flag:
//      0: the cubic spline should be a quadratic over the last interval;
//      1: the first derivative at the right endpoint should be YBCEND;
//      2: the second derivative at the right endpoint should be YBCEND.
//
//    Input, double YBCEND, the values to be used in the boundary
//    conditions if IBCEND is equal to 1 or 2.
//
//    Output, double SPLINE_CUBIC_SET[N], the second derivatives of the cubic spline.
//
{
  double *a;
  double *b;
  int i;
  double *ypp;
//
//  Check.
//
  if ( n <= 1 )
  {
    cout << "\n";
    cout << "SPLINE_CUBIC_SET - Fatal error!\n";
    cout << "  The number of data points N must be at least 2.\n";
    cout << "  The input value is " << n << ".\n";
    return NULL;
  }

  for ( i = 0; i < n - 1; i++ )
  {
    if ( t[i+1] <= t[i] )
    {
      cout << "\n";
      cout << "SPLINE_CUBIC_SET - Fatal error!\n";
      cout << "  The knots must be strictly increasing, but\n";
      cout << "  T(" << i   << ") = " << t[i]   << "\n";
      cout << "  T(" << i+1 << ") = " << t[i+1] << "\n";
      return NULL;
    }
  }
  a = new double[3*n];
  b = new double[n];
//
//  Set up the first equation.
//
  if ( ibcbeg == 0 )
  {
    b[0] = 0.0;
    a[1+0*3] = 1.0;
    a[0+1*3] = -1.0;
  }
  else if ( ibcbeg == 1 )
  {
    b[0] = ( y[1] - y[0] ) / ( t[1] - t[0] ) - ybcbeg;
    a[1+0*3] = ( t[1] - t[0] ) / 3.0;
    a[0+1*3] = ( t[1] - t[0] ) / 6.0;
  }
  else if ( ibcbeg == 2 )
  {
    b[0] = ybcbeg;
    a[1+0*3] = 1.0;
    a[0+1*3] = 0.0;
  }
  else
  {
    cout << "\n";
    cout << "SPLINE_CUBIC_SET - Fatal error!\n";
    cout << "  IBCBEG must be 0, 1 or 2.\n";
    cout << "  The input value is " << ibcbeg << ".\n";
    delete [] a;
    delete [] b;
    return NULL;
  }
//
//  Set up the intermediate equations.
//
  for ( i = 1; i < n-1; i++ )
  {
    b[i] = ( y[i+1] - y[i] ) / ( t[i+1] - t[i] )
      - ( y[i] - y[i-1] ) / ( t[i] - t[i-1] );
    a[2+(i-1)*3] = ( t[i] - t[i-1] ) / 6.0;
    a[1+ i   *3] = ( t[i+1] - t[i-1] ) / 3.0;
    a[0+(i+1)*3] = ( t[i+1] - t[i] ) / 6.0;
  }
//
//  Set up the last equation.
//
  if ( ibcend == 0 )
  {
    b[n-1] = 0.0;
    a[2+(n-2)*3] = -1.0;
    a[1+(n-1)*3] = 1.0;
  }
  else if ( ibcend == 1 )
  {
    b[n-1] = ybcend - ( y[n-1] - y[n-2] ) / ( t[n-1] - t[n-2] );
    a[2+(n-2)*3] = ( t[n-1] - t[n-2] ) / 6.0;
    a[1+(n-1)*3] = ( t[n-1] - t[n-2] ) / 3.0;
  }
  else if ( ibcend == 2 )
  {
    b[n-1] = ybcend;
    a[2+(n-2)*3] = 0.0;
    a[1+(n-1)*3] = 1.0;
  }
  else
  {
    cout << "\n";
    cout << "SPLINE_CUBIC_SET - Fatal error!\n";
    cout << "  IBCEND must be 0, 1 or 2.\n";
    cout << "  The input value is " << ibcend << ".\n";
    delete [] a;
    delete [] b;
    return NULL;
  }
//
//  Solve the linear system.
//
  if ( n == 2 && ibcbeg == 0 && ibcend == 0 )
  {
    ypp = new double[2];

    ypp[0] = 0.0;
    ypp[1] = 0.0;
  }
  else
  {
    ypp = d3_np_fs ( n, a, b );

    if ( !ypp )
    {
      cout << "\n";
      cout << "SPLINE_CUBIC_SET - Fatal error!\n";
      cout << "  The linear system could not be solved.\n";
      delete [] a;
      delete [] b;
      return NULL;
    }

  }

  delete [] a;
  delete [] b;
  return ypp;
}
//****************************************************************************80

double spline_cubic_val ( int n, double t[], double tval, double y[],
  double ypp[], double *ypval, double *yppval )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_CUBIC_VAL evaluates a piecewise cubic spline at a point.
//
//  Discussion:
//
//    SPLINE_CUBIC_SET must have already been called to define the values of YPP.
//
//    For any point T in the interval T(IVAL), T(IVAL+1), the form of
//    the spline is
//
//      SPL(T) = A
//             + B * ( T - T(IVAL) )
//             + C * ( T - T(IVAL) )**2
//             + D * ( T - T(IVAL) )**3
//
//    Here:
//      A = Y(IVAL)
//      B = ( Y(IVAL+1) - Y(IVAL) ) / ( T(IVAL+1) - T(IVAL) )
//        - ( YPP(IVAL+1) + 2 * YPP(IVAL) ) * ( T(IVAL+1) - T(IVAL) ) / 6
//      C = YPP(IVAL) / 2
//      D = ( YPP(IVAL+1) - YPP(IVAL) ) / ( 6 * ( T(IVAL+1) - T(IVAL) ) )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    04 February 1999
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of knots.
//
//    Input, double Y[N], the data values at the knots.
//
//    Input, double T[N], the knot values.
//
//    Input, double TVAL, a point, typically between T[0] and T[N-1], at
//    which the spline is to be evalulated.  If TVAL lies outside
//    this range, extrapolation is used.
//
//    Input, double Y[N], the data values at the knots.
//
//    Input, double YPP[N], the second derivatives of the spline at
//    the knots.
//
//    Output, double *YPVAL, the derivative of the spline at TVAL.
//
//    Output, double *YPPVAL, the second derivative of the spline at TVAL.
//
//    Output, double SPLINE_VAL, the value of the spline at TVAL.
//
{
  double dt;
  double h;
  int i;
  int ival;
  double yval;
//
//  Determine the interval [ T(I), T(I+1) ] that contains TVAL.
//  Values below T[0] or above T[N-1] use extrapolation.
//
  ival = n - 2;

  for ( i = 0; i < n-1; i++ )
  {
    if ( tval < t[i+1] )
    {
      ival = i;
      break;
    }
  }
//
//  In the interval I, the polynomial is in terms of a normalized
//  coordinate between 0 and 1.
//
  dt = tval - t[ival];
  h = t[ival+1] - t[ival];

  yval = y[ival]
    + dt * ( ( y[ival+1] - y[ival] ) / h
           - ( ypp[ival+1] / 6.0 + ypp[ival] / 3.0 ) * h
    + dt * ( 0.5 * ypp[ival]
    + dt * ( ( ypp[ival+1] - ypp[ival] ) / ( 6.0 * h ) ) ) );

  *ypval = ( y[ival+1] - y[ival] ) / h
    - ( ypp[ival+1] / 6.0 + ypp[ival] / 3.0 ) * h
    + dt * ( ypp[ival]
    + dt * ( 0.5 * ( ypp[ival+1] - ypp[ival] ) / h ) );

  *yppval = ypp[ival] + dt * ( ypp[ival+1] - ypp[ival] ) / h;

  return yval;
}
//****************************************************************************80

void spline_cubic_val2 ( int n, double t[], double tval, int *left, double y[],
  double ypp[], double *yval, double *ypval, double *yppval )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_CUBIC_VAL2 evaluates a piecewise cubic spline at a point.
//
//  Discussion:
//
//    This routine is a modification of SPLINE_CUBIC_VAL; it allows the
//    user to speed up the code by suggesting the appropriate T interval
//    to search first.
//
//    SPLINE_CUBIC_SET must have already been called to define the
//    values of YPP.
//
//    In the LEFT interval, let RIGHT = LEFT+1.  The form of the spline is
//
//    SPL(T) =
//      A
//    + B * ( T - T[LEFT] )
//    + C * ( T - T[LEFT] )**2
//    + D * ( T - T[LEFT] )**3
//
//    Here:
//      A = Y[LEFT]
//      B = ( Y[RIGHT] - Y[LEFT] ) / ( T[RIGHT] - T[LEFT] )
//        - ( YPP[RIGHT] + 2 * YPP[LEFT] ) * ( T[RIGHT] - T[LEFT] ) / 6
//      C = YPP[LEFT] / 2
//      D = ( YPP[RIGHT] - YPP[LEFT] ) / ( 6 * ( T[RIGHT] - T[LEFT] ) )
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of knots.
//
//    Input, double T[N], the knot values.
//
//    Input, double TVAL, a point, typically between T[0] and T[N-1], at
//    which the spline is to be evalulated.  If TVAL lies outside
//    this range, extrapolation is used.
//
//    Input/output, int *LEFT, the suggested T interval to search.
//    LEFT should be between 1 and N-1.  If LEFT is not in this range,
//    then its value will be ignored.  On output, LEFT is set to the
//    actual interval in which TVAL lies.
//
//    Input, double Y[N], the data values at the knots.
//
//    Input, double YPP[N], the second derivatives of the spline at
//    the knots.
//
//    Output, double *YVAL, *YPVAL, *YPPVAL, the value of the spline, and
//    its first two derivatives at TVAL.
//
{
  double dt;
  double h;
  int right;
//
//  Determine the interval [T[LEFT], T[RIGHT]] that contains TVAL.
//
//  What you want from R8VEC_BRACKET3 is that TVAL is to be computed
//  by the data in interval [T[LEFT-1], T[RIGHT-1]].
//
  r8vec_bracket3 ( n, t, tval, left );
//
// In the interval LEFT, the polynomial is in terms of a normalized
// coordinate  ( DT / H ) between 0 and 1.
//
  right = *left + 1;

  dt = tval - t[*left-1];
  h = t[right-1] - t[*left-1];

  *yval = y[*left-1]
     + dt * ( ( y[right-1] - y[*left-1] ) / h
        - ( ypp[right-1] / 6.0 + ypp[*left-1] / 3.0 ) * h
     + dt * ( 0.5 * ypp[*left-1]
     + dt * ( ( ypp[right-1] - ypp[*left-1] ) / ( 6.0 * h ) ) ) );

  *ypval = ( y[right-1] - y[*left-1] ) / h
     - ( ypp[right-1] / 6.0 + ypp[*left-1] / 3.0 ) * h
     + dt * ( ypp[*left-1]
     + dt * ( 0.5 * ( ypp[right-1] - ypp[*left-1] ) / h ) );

  *yppval = ypp[*left-1] + dt * ( ypp[right-1] - ypp[*left-1] ) / h;

  return;
}
//****************************************************************************80

double *spline_hermite_set ( int ndata, double tdata[], double ydata[],
  double ypdata[] )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_HERMITE_SET sets up a piecewise cubic Hermite interpolant.
//
//  Discussion:
//
//    Once the array C is computed, then in the interval
//    (TDATA(I), TDATA(I+1)), the interpolating Hermite polynomial
//    is given by
//
//      SVAL(TVAL) =                 C(1,I)
//         + ( TVAL - TDATA(I) ) * ( C(2,I)
//         + ( TVAL - TDATA(I) ) * ( C(3,I)
//         + ( TVAL - TDATA(I) ) *   C(4,I) ) )
//
//    This is algorithm CALCCF from Conte and deBoor.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Samuel Conte, Carl deBoor,
//    Elementary Numerical Analysis,
//    Second Edition,
//    McGraw Hill, 1972,
//    ISBN: 07-012446-4.
//
//  Parameters:
//
//    Input, int NDATA, the number of data points.
//    NDATA must be at least 2.
//
//    Input, double TDATA[NDATA], the abscissas of the data points.
//    The entries of TDATA are assumed to be strictly increasing.
//
//    Input, double Y[NDATA], YP[NDATA], the value of the
//    function and its derivative at TDATA(1:NDATA).
//
//    Output, double SPLINE_HERMITE_SET[4*NDATA], the coefficients of
//    the Hermite polynomial.  We will refer to this array as "C".
//    C(1,1:NDATA) = Y(1:NDATA) and C(2,1:NDATA) = YP(1:NDATA).
//    C(3,1:NDATA-1) and C(4,1:NDATA-1) are the quadratic and cubic
//    coefficients.
//
{
  double *c;
  double divdif1;
  double divdif3;
  double dt;
  int i;
  int j;

  c = new double[4*ndata];

  for ( j = 0; j < ndata; j++ )
  {
    c[0+j*4] = ydata[j];
  }

  for ( j = 0; j < ndata; j++ )
  {
    c[1+j*4] = ypdata[j];
  }

  for ( i = 1; i <= ndata-1; i++ )
  {
    dt = tdata[i] - tdata[i-1];
    divdif1 = ( c[0+i*4] - c[0+(i-1)*4] ) / dt;
    divdif3 = c[1+(i-1)*4] + c[1+i*4] - 2.0 * divdif1;
    c[2+(i-1)*4] = ( divdif1 - c[1+(i-1)*4] - divdif3 ) / dt;
    c[3+(i-1)*4] = divdif3 / ( dt * dt );
  }

  c[2+(ndata-1)*4] = 0.0;
  c[3+(ndata-1)*4] = 0.0;

  return c;
}
//****************************************************************************80

void spline_hermite_val ( int ndata, double tdata[], double c[], double tval,
  double *sval, double *spval )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_HERMITE_VAL evaluates a piecewise cubic Hermite interpolant.
//
//  Discussion:
//
//    SPLINE_HERMITE_SET must be called first, to set up the
//    spline data from the raw function and derivative data.
//
//    In the interval (TDATA(I), TDATA(I+1)), the interpolating
//    Hermite polynomial is given by
//
//      SVAL(TVAL) =                 C(1,I)
//         + ( TVAL - TDATA(I) ) * ( C(2,I)
//         + ( TVAL - TDATA(I) ) * ( C(3,I)
//         + ( TVAL - TDATA(I) ) *   C(4,I) ) )
//
//    and
//
//      SVAL'(TVAL) =                    C(2,I)
//         + ( TVAL - TDATA(I) ) * ( 2 * C(3,I)
//         + ( TVAL - TDATA(I) ) *   3 * C(4,I) )
//
//    This is algorithm PCUBIC from Conte and deBoor.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    Samuel Conte, Carl deBoor,
//    Elementary Numerical Analysis,
//    Second Edition,
//    McGraw Hill, 1972,
//    ISBN: 07-012446-4.
//
//  Parameters:
//
//    Input, int NDATA, the number of data points.
//    NDATA is assumed to be at least 2.
//
//    Input, double TDATA[NDATA], the abscissas of the data points.
//    The entries of TDATA are assumed to be strictly increasing.
//
//    Input, double C[4*NDATA], the coefficient data computed by
//    SPLINE_HERMITE_SET.
//
//    Input, double TVAL, the point where the interpolant is to
//    be evaluated.
//
//    Output, double *SVAL, *SPVAL, the value of the interpolant
//    and its derivative at TVAL.
//
{
  double dt;
  int left;
  int right;
//
//  Find the interval [ TDATA(LEFT), TDATA(RIGHT) ] that contains
//  or is nearest to TVAL.
//
  r8vec_bracket ( ndata, tdata, tval, &left, &right );
//
//  Evaluate the cubic polynomial.
//
  dt = tval - tdata[left-1];

  *sval =        c[0+(left-1)*4]
        + dt * ( c[1+(left-1)*4]
        + dt * ( c[2+(left-1)*4]
        + dt *   c[3+(left-1)*4] ) );

  *spval =             c[1+(left-1)*4]
        + dt * ( 2.0 * c[2+(left-1)*4]
        + dt *   3.0 * c[3+(left-1)*4] );

  return;
}
//****************************************************************************80

double spline_linear_int ( int ndata, double tdata[], double ydata[],
  double a, double b )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_LINEAR_INT evaluates the integral of a piecewise linear spline.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    25 January 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int NDATA, the number of data points defining the spline.
//
//    Input, double TDATA[NDATA], YDATA[NDATA], the values of the independent
//    and dependent variables at the data points.  The values of TDATA should
//    be distinct and increasing.
//
//    Input, double A, B, the interval over which the integral is desired.
//
//    Output, double SPLINE_LINEAR_INT, the value of the integral.
//
{
  double a_copy;
  int a_left;
  int a_right;
  double b_copy;
  int b_left;
  int b_right;
  int i_left;
  double int_val;
  double tval;
  double yp;
  double yval;

  int_val = 0.0;

  if ( a == b )
  {
    return int_val;
  }

  a_copy = r8_min ( a, b );
  b_copy = r8_max ( a, b );
//
//  Find the interval [ TDATA(A_LEFT), TDATA(A_RIGHT) ] that contains, or is
//  nearest to, A.
//
  r8vec_bracket ( ndata, tdata, a_copy, &a_left, &a_right );
//
//  Find the interval [ TDATA(B_LEFT), TDATA(B_RIGHT) ] that contains, or is
//  nearest to, B.
//
  r8vec_bracket ( ndata, tdata, b_copy, &b_left, &b_right );
//
//  If A and B are in the same interval...
//
  if ( a_left == b_left )
  {
    tval = ( a_copy + b_copy ) / 2.0;

    yp = ( ydata[a_right-1] - ydata[a_left-1] ) /
         ( tdata[a_right-1] - tdata[a_left-1] );

    yval = ydata[a_left-1] + ( tval - tdata[a_left-1] ) * yp;

    int_val = yval * ( b_copy - a_copy );

    return int_val;
  }
//
//  Otherwise, integrate from:
//
//  A               to TDATA(A_RIGHT),
//  TDATA(A_RIGHT)  to TDATA(A_RIGHT+1),...
//  TDATA(B_LEFT-1) to TDATA(B_LEFT),
//  TDATA(B_LEFT)   to B.
//
//  Use the fact that the integral of a linear function is the
//  value of the function at the midpoint times the width of the interval.
//
  tval = ( a_copy + tdata[a_right-1] ) / 2.0;

  yp = ( ydata[a_right-1] - ydata[a_left-1] ) /
       ( tdata[a_right-1] - tdata[a_left-1] );

  yval = ydata[a_left-1] + ( tval - tdata[a_left-1] ) * yp;

  int_val = int_val + yval * ( tdata[a_right-1] - a_copy );

  for ( i_left = a_right; i_left <= b_left - 1; i_left++ )
  {
    tval = ( tdata[i_left] + tdata[i_left-1] ) / 2.0;

    yp = ( ydata[i_left-1] - ydata[i_left-2] ) /
         ( tdata[i_left-1] - tdata[i_left-2] );

    yval = ydata[i_left-2] + ( tval - tdata[i_left-2] ) * yp;

    int_val = int_val + yval * ( tdata[i_left-1] - tdata[i_left-2] );
  }

  tval = ( tdata[b_left-1] + b_copy ) / 2.0E+0;

  yp = ( ydata[b_right-1] - ydata[b_left-1] ) /
       ( tdata[b_right-1] - tdata[b_left-1] );

  yval = ydata[b_left-1] + ( tval - tdata[b_left-1] ) * yp;

  int_val = int_val + yval * ( b_copy - tdata[b_left-1] );

  if ( b < a )
  {
    int_val = -int_val;
  }

  return int_val;
}
//****************************************************************************80

void spline_linear_intset ( int n, double int_x[], double int_v[],
  double data_x[], double data_y[] )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_LINEAR_INTSET sets a piecewise linear spline with given integral properties.
//
//  Discussion:
//
//    The user has in mind an interval, divided by N+1 points into
//    N intervals.  A linear spline is to be constructed,
//    with breakpoints at the centers of each interval, and extending
//    continuously to the left of the first and right of the last
//    breakpoints.  The constraint on the linear spline is that it is
//    required that it have a given integral value over each interval.
//
//    A tridiagonal linear system of equations is solved for the
//    values of the spline at the breakpoints.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    07 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the number of intervals.
//
//    Input, double INT_X[N+1], the points that define the intervals.
//    Interval I lies between INT_X(I) and INT_X(I+1).
//
//    Input, double INT_V[N], the desired value of the integral of the
//    linear spline over each interval.
//
//    Output, double DATA_X[N], DATA_Y[N], the values of the independent
//    and dependent variables at the data points.  The values of DATA_X are
//    the interval midpoints.  The values of DATA_Y are determined in such
//    a way that the exact integral of the linear spline over interval I
//    is equal to INT_V(I).
//
{
  double *a;
  double *b;
  double *c;
  int i;

  a = new double[3*n];
  b = new double[n];
//
//  Set up the easy stuff.
//
  for ( i = 1; i <= n; i++ )
  {
    data_x[i-1] = 0.5 * ( int_x[i-1] + int_x[i] );
  }
//
//  Set up the coefficients of the linear system.
//
  for ( i = 0; i < n-2; i++ )
  {
    a[2+i*3] = 1.0 - ( 0.5 * ( data_x[i+1] + int_x[i+1] )
      - data_x[i] ) / ( data_x[i+1] - data_x[i] );
  }
  a[2+(n-2)*3] = 0.0;
  a[2+(n-1)*3] = 0.0;

  a[1+0*3] = int_x[1] - int_x[0];

  for ( i = 1; i < n-1; i++ )
  {
    a[1+i*3] = 1.0 + ( 0.5 * ( data_x[i] + int_x[i] )
    - data_x[i-1] ) / ( data_x[i] - data_x[i-1] )
    - ( 0.5 * ( data_x[i] + int_x[i+1] ) - data_x[i] )
    / ( data_x[i+1] - data_x[i] );
  }
  a[1+(n-1)*3] = int_x[n] - int_x[n-1];

  a[0+0*3] = 0.0;
  a[0+1*3] = 0.0;
  for ( i = 2; i < n; i++ )
  {
    a[0+i*3] = ( 0.5 * ( data_x[i-1] + int_x[i] )
    - data_x[i-1] ) / ( data_x[i] - data_x[i-1] );
  }
//
//  Set up DATA_Y, which begins as the right hand side of the linear system.
//
  b[0] = int_v[0];
  for ( i = 2; i <= n-1; i++ )
  {
    b[i-1] = 2.0 * int_v[i-1] / ( int_x[i] - int_x[i-1] );
  }
  b[n-1] = int_v[n-1];
//
//  Solve the linear system.
//
  c = d3_np_fs ( n, a, b );

  for ( i = 0; i < n; i++ )
  {
     data_y[i] = c[i];
  }

  delete [] a;
  delete [] b;
  delete [] c;

  return;
}
//****************************************************************************80

void spline_linear_val ( int ndata, double tdata[], double ydata[],
  double tval, double *yval, double *ypval )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_LINEAR_VAL evaluates a piecewise linear spline at a point.
//
//  Discussion:
//
//    Because of the extremely simple form of the linear spline,
//    the raw data points ( TDATA(I), YDATA(I)) can be used directly to
//    evaluate the spline at any point.  No processing of the data
//    is required.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int NDATA, the number of data points defining the spline.
//
//    Input, double TDATA[NDATA], YDATA[NDATA], the values of the independent
//    and dependent variables at the data points.  The values of TDATA should
//    be distinct and increasing.
//
//    Input, double TVAL, the point at which the spline is to be evaluated.
//
//    Output, double *YVAL, *YPVAL, the value of the spline and its first
//    derivative dYdT at TVAL.  YPVAL is not reliable if TVAL is exactly
//    equal to TDATA(I) for some I.
//
{
  int left;
  int right;
//
//  Find the interval [ TDATA(LEFT), TDATA(RIGHT) ] that contains, or is
//  nearest to, TVAL.
//
  r8vec_bracket ( ndata, tdata, tval, &left, &right );
//
//  Now evaluate the piecewise linear function.
//
  *ypval = ( ydata[right-1] - ydata[left-1] )
         / ( tdata[right-1] - tdata[left-1] );

  *yval = ydata[left-1] +  ( tval - tdata[left-1] ) * (*ypval);

  return;
}
//****************************************************************************80

double spline_overhauser_nonuni_val ( int ndata, double tdata[],
  double ydata[], double tval )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_OVERHAUSER_NONUNI_VAL evaluates the nonuniform Overhauser spline.
//
//  Discussion:
//
//    The nonuniformity refers to the fact that the abscissas values
//    need not be uniformly spaced.
//
//    Thanks to Doug Fortune for pointing out that the point distances
//    used to define ALPHA and BETA should be the Euclidean distances
//    between the points.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    08 May 2007
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int NDATA, the number of data points.
//    NDATA must be at least 3.
//
//    Input, double TDATA[NDATA], the abscissas of the data points.
//    The values of TDATA are assumed to be distinct and increasing.
//
//    Input, double YDATA[NDATA], the data values.
//
//    Input, double TVAL, the value where the spline is to
//    be evaluated.
//
//    Output, double SPLINE_OVERHAUSER_NONUNI_VAL, the value of the
//    spline at TVAL.
//
{
  double alpha;
  double beta;
  double d21;
  double d32;
  double d43;
  int left;
  double *mbasis;
  int right;
  double yval;
//
//  Check NDATA.
//
  if ( ndata < 3 )
  {
    cout << "\n";
    cout << "SPLINE_OVERHAUSER_NONUNI_VAL - Fatal error!\n";
    cout << "  NDATA < 3.\n";
    exit ( 1 );
  }
//
//  Find the nearest interval [ TDATA(LEFT), TDATA(RIGHT) ] to TVAL.
//
  r8vec_bracket ( ndata, tdata, tval, &left, &right );
//
//  Evaluate the spline in the given interval.
//
  if ( left == 1 )
  {
    d21 = sqrt ( pow ( tdata[1] - tdata[0], 2 )
               + pow ( ydata[1] - ydata[0], 2 ) );

    d32 = sqrt ( pow ( tdata[2] - tdata[1], 2 )
               + pow ( ydata[2] - ydata[1], 2 ) );

    alpha = d21 / ( d32 + d21 );

    mbasis = basis_matrix_overhauser_nul ( alpha );

    yval = basis_matrix_tmp ( left, 3, mbasis, ndata, tdata, ydata, tval );
  }
  else if ( left < ndata-1 )
  {
    d21 = sqrt ( pow ( tdata[left-1] - tdata[left-2], 2 )
               + pow ( ydata[left-1] - ydata[left-2], 2 ) );

    d32 = sqrt ( pow ( tdata[left] - tdata[left-1], 2 )
               + pow ( ydata[left] - ydata[left-1], 2 ) );

    d43 = sqrt ( pow ( tdata[left+1] - tdata[left], 2 )
               + pow ( ydata[left+1] - ydata[left], 2 ) );

    alpha = d21 / ( d32 + d21 );
    beta  = d32 / ( d43 + d32 );

    mbasis = basis_matrix_overhauser_nonuni ( alpha, beta );

    yval = basis_matrix_tmp ( left, 4, mbasis, ndata, tdata, ydata, tval );
  }
  else if ( left == ndata-1 )
  {
    d32 = sqrt ( pow ( tdata[ndata-2] - tdata[ndata-3], 2 )
               + pow ( ydata[ndata-2] - ydata[ndata-3], 2 ) );

    d43 = sqrt ( pow ( tdata[ndata-1] - tdata[ndata-2], 2 )
               + pow ( ydata[ndata-1] - ydata[ndata-2], 2 ) );

    beta  = d32 / ( d43 + d32 );

    mbasis = basis_matrix_overhauser_nur ( beta );

    yval = basis_matrix_tmp ( left, 3, mbasis, ndata, tdata, ydata, tval );
  }
  else
  {
    cout << "\n";
    cout << "SPLINE_OVERHAUSER_NONUNI_VAL - Fatal error!\n";
    cout << "  Nonsensical value of LEFT = " << left << "\n";
    cout << "  but 0 < LEFT < NDATA = " << ndata << "\n";
    cout << "  is required.\n";
    exit ( 1 );
  }

  delete [] mbasis;

  return yval;
}
//****************************************************************************80

double spline_overhauser_uni_val ( int ndata, double tdata[], double ydata[],
  double tval )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_OVERHAUSER_UNI_VAL evaluates the uniform Overhauser spline.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int NDATA, the number of data points.
//    NDATA must be at least 3.
//
//    Input, double TDATA[NDATA], the abscissas of the data points.
//    The values of TDATA are assumed to be distinct and increasing.
//    This routine also assumes that the values of TDATA are uniformly
//    spaced; for instance, TDATA(1) = 10, TDATA(2) = 11, TDATA(3) = 12...
//
//    Input, double YDATA[NDATA], the data values.
//
//    Input, double TVAL, the value where the spline is to
//    be evaluated.
//
//    Output, double SPLINE_OVERHAUSER_UNI_VAL, the value of the spline at TVAL.
//
{
  int left=0;
  double *mbasis=NULL;
  int right=0;
  double yval=0.0;
//
//  Check NDATA.
//
  if ( ndata < 3 )
  {
    cout << "\n";
    cout << "SPLINE_OVERHAUSER_UNI_VAL - Fatal error!\n";
    cout << "  NDATA < 3.\n";
    exit ( 1 );
  }
//
//  Find the nearest interval [ TDATA(LEFT), TDATA(RIGHT) ] to TVAL.
//
  r8vec_bracket ( ndata, tdata, tval, &left, &right );
//
//  Evaluate the spline in the given interval.
//
  if ( left == 1 )
  {

    mbasis = basis_matrix_overhauser_uni_l ( );

    yval = basis_matrix_tmp ( left, 3, mbasis, ndata, tdata, ydata, tval );
  }
  else if ( left < ndata-1 )
  {
    mbasis = basis_matrix_overhauser_uni ( );

    yval = basis_matrix_tmp ( left, 4, mbasis, ndata, tdata, ydata, tval );
  }
  else if ( left == ndata-1 )
  {
    mbasis = basis_matrix_overhauser_uni_r ( );

    yval = basis_matrix_tmp ( left, 3, mbasis, ndata, tdata, ydata, tval );

  }

  delete [] mbasis;

  return yval;
}
//****************************************************************************80

void spline_overhauser_val ( int ndim, int ndata, double tdata[],
  double ydata[], double tval, double yval[] )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_OVERHAUSER_VAL evaluates an Overhauser spline.
//
//  Discussion:
//
//    Over the first and last intervals, the Overhauser spline is a
//    quadratic.  In the intermediate intervals, it is a piecewise cubic.
//    The Overhauser spline is also known as the Catmull-Rom spline.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    11 December 2004
//
//  Author:
//
//    John Burkardt
//
//  Reference:
//
//    JA Brewer, DC Anderson,
//    Visual Interaction with Overhauser Curves and Surfaces,
//    SIGGRAPH 77,
//    in Proceedings of the 4th Annual Conference on Computer Graphics
//    and Interactive Techniques,
//    ASME, July 1977, pages 132-137.
//
//    Edwin Catmull, Raphael Rom,
//    A Class of Local Interpolating Splines,
//    in Computer Aided Geometric Design,
//    edited by Robert Barnhill, Richard Reisenfeld,
//    Academic Press, 1974,
//    ISBN: 0120790505.
//
//    David Rogers, Alan Adams,
//    Mathematical Elements of Computer Graphics,
//    Second Edition,
//    McGraw Hill, 1989,
//    ISBN: 0070535299.
//
//  Parameters:
//
//    Input, int NDIM, the dimension of a single data point.
//    NDIM must be at least 1.
//
//    Input, int NDATA, the number of data points.
//    NDATA must be at least 3.
//
//    Input, double TDATA[NDATA], the abscissas of the data points.  The
//    values in TDATA must be in strictly ascending order.
//
//    Input, double YDATA[NDIM*NDATA], the data points corresponding to
//    the abscissas.
//
//    Input, double TVAL, the abscissa value at which the spline
//    is to be evaluated.  Normally, TDATA[0] <= TVAL <= T[NDATA-1], and
//    the data will be interpolated.  For TVAL outside this range,
//    extrapolation will be used.
//
//    Output, double YVAL[NDIM], the value of the spline at TVAL.
//
{
  int i;
  int left;
  int order;
  int right;
  double *yl;
  double *yr;
//
//  Check.
//
  r8vec_order_type ( ndata, tdata, &order );

  if ( order != 2 )
  {
    cout << "\n";
    cout << "SPLINE_OVERHAUSER_VAL - Fatal error!\n";
    cout << "  The data abscissas are not strictly ascending.\n";
    exit ( 1 );
  }

  if ( ndata < 3 )
  {
    cout << "\n";
    cout << "SPLINE_OVERHAUSER_VAL - Fatal error!\n";
    cout << "  NDATA < 3.\n";
    exit ( 1 );
  }
//
//  Locate the abscissa interval T[LEFT], T[LEFT+1] nearest to or
//  containing TVAL.
//
  r8vec_bracket ( ndata, tdata, tval, &left, &right );
//
//  Evaluate the "left hand" quadratic defined at
//  T[LEFT-1], T[LEFT], T[RIGHT].
//
  yl = new double[ndim];
  yr = new double[ndim];

  if ( 0 < left-1 )
  {
    parabola_val2 ( ndim, ndata, tdata, ydata, left-1, tval, yl );
  }
//
//  Evaluate the "right hand" quadratic defined at
//  T[LEFT], T[RIGHT], T[RIGHT+1].
//
  if ( right+1 <= ndata )
  {
    parabola_val2 ( ndim, ndata, tdata, ydata, left, tval, yr );
  }
//
//  Blend the quadratics.
//
  if ( left == 1 )
  {
    for ( i = 0; i < ndim; i++ )
    {
      yval[i] = yr[i];
    }
  }
  else if ( right < ndata )
  {
    for ( i = 0; i < ndim; i++ )
    {
      yval[i] = (
          ( tdata[right-1] - tval                 ) * yl[i]
        + (                  tval - tdata[left-1] ) * yr[i] )
        / ( tdata[right-1]        - tdata[left-1] );
    }
  }
  else
  {
    for ( i = 0; i < ndim; i++ )
    {
      yval[i] = yl[i];
    }
  }

  delete [] yl;
  delete [] yr;

  return;
}
//****************************************************************************80

void spline_pchip_set ( int n, double x[], double f[], double d[] )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_PCHIP_SET sets derivatives for a piecewise cubic Hermite interpolant.
//
//  Discussion:
//
//    This routine computes what would normally be called a Hermite
//    interpolant.  However, the user is only required to supply function
//    values, not derivative values as well.  This routine computes
//    "suitable" derivative values, so that the resulting Hermite interpolant
//    has desirable shape and monotonicity properties.
//
//    The interpolant will have an extremum at each point where
//    monotonicity switches direction.
//
//    The resulting piecewise cubic Hermite function may be evaluated
//    by SPLINE_PCHIP_VAL..
//
//    This routine was originally called "PCHIM".
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 August 2005
//
//  Author:
//
//    FORTRAN77 original version by Fred Fritsch, Lawrence Livermore National Laboratory.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Fred Fritsch, Ralph Carlson,
//    Monotone Piecewise Cubic Interpolation,
//    SIAM Journal on Numerical Analysis,
//    Volume 17, Number 2, April 1980, pages 238-246.
//
//    Fred Fritsch, Judy Butland,
//    A Method for Constructing Local Monotone Piecewise
//    Cubic Interpolants,
//    SIAM Journal on Scientific and Statistical Computing,
//    Volume 5, Number 2, 1984, pages 300-304.
//
//  Parameters:
//
//    Input, int N, the number of data points.  N must be at least 2.
//
//    Input, double X[N], the strictly increasing independent
//    variable values.
//
//    Input, double F[N], dependent variable values to be interpolated.  This
//    routine is designed for monotonic data, but it will work for any F-array.
//    It will force extrema at points where monotonicity switches direction.
//
//    Output, double D[N], the derivative values at the
//    data points.  If the data are monotonic, these values will determine
//    a monotone cubic Hermite function.
//
{
  double del1;
  double del2;
  double dmax;
  double dmin;
  double drat1;
  double drat2;
  double dsave;
  double h1;
  double h2;
  double hsum;
  double hsumt3;
  int i;
  int ierr;
  int nless1;
  double temp;
  double w1;
  double w2;
//
//  Check the arguments.
//
  if ( n < 2 )
  {
    ierr = -1;
    cout << "\n";
    cout << "SPLINE_PCHIP_SET - Fatal error!\n";
    cout << "  Number of data points less than 2.\n";
    exit ( ierr );
  }

  for ( i = 1; i < n; i++ )
  {
    if ( x[i] <= x[i-1] )
    {
      ierr = -3;
      cout << "\n";
      cout << "SPLINE_PCHIP_SET - Fatal error!\n";
      cout << "  X array not strictly increasing.\n";
      exit ( ierr );
    }
  }

  ierr = 0;
  nless1 = n - 1;
  h1 = x[1] - x[0];
  del1 = ( f[1] - f[0] ) / h1;
  dsave = del1;
//
//  Special case N=2, use linear interpolation.
//
  if ( n == 2 )
  {
    d[0] = del1;
    d[n-1] = del1;
    return;
  }
//
//  Normal case, 3 <= N.
//
  h2 = x[2] - x[1];
  del2 = ( f[2] - f[1] ) / h2;
//
//  Set D(1) via non-centered three point formula, adjusted to be
//  shape preserving.
//
  hsum = h1 + h2;
  w1 = ( h1 + hsum ) / hsum;
  w2 = -h1 / hsum;
  d[0] = w1 * del1 + w2 * del2;

  if ( pchst ( d[0], del1 ) <= 0.0 )
  {
    d[0] = 0.0;
  }
//
//  Need do this check only if monotonicity switches.
//
  else if ( pchst ( del1, del2 ) < 0.0 )
  {
     dmax = 3.0 * del1;

     if ( fabs ( dmax ) < fabs ( d[0] ) )
     {
       d[0] = dmax;
     }

  }
//
//  Loop through interior points.
//
  for ( i = 2; i <= nless1; i++ )
  {
    if ( 2 < i )
    {
      h1 = h2;
      h2 = x[i] - x[i-1];
      hsum = h1 + h2;
      del1 = del2;
      del2 = ( f[i] - f[i-1] ) / h2;
    }
//
//  Set D(I)=0 unless data are strictly monotonic.
//
    d[i-1] = 0.0;

    temp = pchst ( del1, del2 );

    if ( temp < 0.0 )
    {
      ierr = ierr + 1;
      dsave = del2;
    }
//
//  Count number of changes in direction of monotonicity.
//
    else if ( temp == 0.0 )
    {
      if ( del2 != 0.0 )
      {
        if ( pchst ( dsave, del2 ) < 0.0 )
        {
          ierr = ierr + 1;
        }
        dsave = del2;
      }
    }
//
//  Use Brodlie modification of Butland formula.
//
    else
    {
      hsumt3 = 3.0 * hsum;
      w1 = ( hsum + h1 ) / hsumt3;
      w2 = ( hsum + h2 ) / hsumt3;
      dmax = r8_max ( fabs ( del1 ), fabs ( del2 ) );
      dmin = r8_min ( fabs ( del1 ), fabs ( del2 ) );
      drat1 = del1 / dmax;
      drat2 = del2 / dmax;
      d[i-1] = dmin / ( w1 * drat1 + w2 * drat2 );
    }
  }
//
//  Set D(N) via non-centered three point formula, adjusted to be
//  shape preserving.
//
  w1 = -h2 / hsum;
  w2 = ( h2 + hsum ) / hsum;
  d[n-1] = w1 * del1 + w2 * del2;

  if ( pchst ( d[n-1], del2 ) <= 0.0 )
  {
    d[n-1] = 0.0;
  }
  else if ( pchst ( del1, del2 ) < 0.0 )
  {
//
//  Need do this check only if monotonicity switches.
//
    dmax = 3.0 * del2;

    if ( fabs ( dmax ) < abs ( d[n-1] ) )
    {
      d[n-1] = dmax;
    }

  }
  return;
}
//****************************************************************************80

void spline_pchip_val ( int n, double x[], double f[], double d[],
  int ne, double xe[], double fe[] )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_PCHIP_VAL evaluates a piecewise cubic Hermite function.
//
//  Description:
//
//    This routine may be used by itself for Hermite interpolation, or as an
//    evaluator for SPLINE_PCHIP_SET.
//
//    This routine evaluates the cubic Hermite function at the points XE.
//
//    Most of the coding between the call to CHFEV and the end of
//    the IR loop could be eliminated if it were permissible to
//    assume that XE is ordered relative to X.
//
//    CHFEV does not assume that X1 is less than X2.  Thus, it would
//    be possible to write a version of SPLINE_PCHIP_VAL that assumes a strictly
//    decreasing X array by simply running the IR loop backwards
//    and reversing the order of appropriate tests.
//
//    The present code has a minor bug, which I have decided is not
//    worth the effort that would be required to fix it.
//    If XE contains points in [X(N-1),X(N)], followed by points less than
//    X(N-1), followed by points greater than X(N), the extrapolation points
//    will be counted (at least) twice in the total returned in IERR.
//
//    The evaluation will be most efficient if the elements of XE are
//    increasing relative to X; that is, for all J <= K,
//      X(I) <= XE(J)
//    implies
//      X(I) <= XE(K).
//
//    If any of the XE are outside the interval [X(1),X(N)],
//    values are extrapolated from the nearest extreme cubic,
//    and a warning error is returned.
//
//    This routine was originally named "PCHFE".
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    14 August 2005
//
//  Author:
//
//    Original FORTRAN77 version by Fred Fritsch, Lawrence Livermore National Laboratory.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    Fred Fritsch, Ralph Carlson,
//    Monotone Piecewise Cubic Interpolation,
//    SIAM Journal on Numerical Analysis,
//    Volume 17, Number 2, April 1980, pages 238-246.
//
//  Parameters:
//
//    Input, int N, the number of data points.  N must be at least 2.
//
//    Input, double X[N], the strictly increasing independent
//    variable values.
//
//    Input, double F[N], the function values.
//
//    Input, double D[N], the derivative values.
//
//    Input, int NE, the number of evaluation points.
//
//    Input, double XE[NE], points at which the function is to
//    be evaluated.
//
//    Output, double FE[NE], the values of the cubic Hermite
//    function at XE.
//
{
  int i;
  int ierc;
  int ierr;
  int ir;
  int j;
  int j_first;
  int j_new;
  int j_save;
  int next[2];
  int nj;
//
//  Check arguments.
//
  if ( n < 2 )
  {
    ierr = -1;
    cout << "\n";
    cout << "SPLINE_PCHIP_VAL - Fatal error!\n";
    cout << "  Number of data points less than 2.\n";
    exit ( ierr );
  }

  for ( i = 1; i < n; i++ )
  {
    if ( x[i] <= x[i-1] )
    {
      ierr = -3;
      cout << "\n";
      cout << "SPLINE_PCHIP_VAL - Fatal error!\n";
      cout << "  X array not strictly increasing.\n";
      exit ( ierr );
    }
  }

  if ( ne < 1 )
  {
    ierr = -4;
    cout << "\n";
    cout << "SPLINE_PCHIP_VAL - Fatal error!\n";
    cout << "  Number of evaluation points less than 1.\n";
    return;
  }

  ierr = 0;
//
//  Loop over intervals.
//  The interval index is IL = IR-1.
//  The interval is X(IL) <= X < X(IR).
//
  j_first = 1;
  ir = 2;

  for ( ; ; )
  {
//
//  Skip out of the loop if have processed all evaluation points.
//
    if ( ne < j_first )
    {
      break;
    }
//
//  Locate all points in the interval.
//
    j_save = ne + 1;

    for ( j = j_first; j <= ne; j++ )
    {
      if ( x[ir-1] <= xe[j-1] )
      {
        j_save = j;
        if ( ir == n )
        {
          j_save = ne + 1;
        }
        break;
      }
    }
//
//  Have located first point beyond interval.
//
    j = j_save;

    nj = j - j_first;
//
//  Skip evaluation if no points in interval.
//
    if ( nj != 0 )
    {
//
//  Evaluate cubic at XE(J_FIRST:J-1).
//
      ierc = chfev ( x[ir-2], x[ir-1], f[ir-2], f[ir-1], d[ir-2], d[ir-1],
        nj, xe+j_first-1, fe+j_first-1, next );

      if ( ierc < 0 )
      {
        ierr = -5;
        cout << "\n";
        cout << "SPLINE_PCHIP_VAL - Fatal error!\n";
        cout << "  Error return from CHFEV.\n";
        exit ( ierr );
      }
//
//  In the current set of XE points, there are NEXT(2) to the right of X(IR).
//
      if ( next[1] != 0 )
      {
        if ( ir < n )
        {
          ierr = -5;
          cout << "\n";
          cout << "SPLINE_PCHIP_VAL - Fatal error!\n";
          cout << "  IR < N.\n";
          exit ( ierr );
        }
//
//  These are actually extrapolation points.
//
        ierr = ierr + next[1];

      }
//
//  In the current set of XE points, there are NEXT(1) to the left of X(IR-1).
//
      if ( next[0] != 0 )
      {
//
//  These are actually extrapolation points.
//
        if ( ir <= 2 )
        {
          ierr = ierr + next[0];
        }
        else
        {
          j_new = -1;

          for ( i = j_first; i <= j-1; i++ )
          {
            if ( xe[i-1] < x[ir-2] )
            {
              j_new = i;
              break;
            }
          }

          if ( j_new == -1 )
          {
            ierr = -5;
            cout << "\n";
            cout << "SPLINE_PCHIP_VAL - Fatal error!\n";
            cout << "  Could not bracket the data point.\n";
            exit ( ierr );
          }
//
//  Reset J.  This will be the new J_FIRST.
//
          j = j_new;
//
//  Now find out how far to back up in the X array.
//
          for ( i = 1; i <= ir-1; i++ )
          {
            if ( xe[j-1] < x[i-1] )
            {
              break;
            }
          }
//
//  At this point, either XE(J) < X(1) or X(i-1) <= XE(J) < X(I) .
//
//  Reset IR, recognizing that it will be incremented before cycling.
//
          ir = i4_max ( 1, i-1 );
        }
      }

      j_first = j;
    }

    ir = ir + 1;

    if ( n < ir )
    {
      break;
    }

  }

  return;
}
//****************************************************************************80

void spline_quadratic_val ( int ndata, double tdata[], double ydata[],
  double tval, double *yval, double *ypval )

//****************************************************************************80
//
//  Purpose:
//
//    SPLINE_QUADRATIC_VAL evaluates a piecewise quadratic spline at a point.
//
//  Discussion:
//
//    Because of the simple form of a piecewise quadratic spline,
//    the raw data points ( TDATA(I), YDATA(I)) can be used directly to
//    evaluate the spline at any point.  No processing of the data
//    is required.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    24 February 2004
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int NDATA, the number of data points defining the spline.
//    NDATA should be odd and at least 3.
//
//    Input, double TDATA[NDATA], YDATA[NDATA], the values of the independent
//    and dependent variables at the data points.  The values of TDATA should
//    be distinct and increasing.
//
//    Input, double TVAL, the point at which the spline is to be evaluated.
//
//    Output, double *YVAL, *YPVAL, the value of the spline and its first
//    derivative dYdT at TVAL.  YPVAL is not reliable if TVAL is exactly
//    equal to TDATA(I) for some I.
//
{
  double dif1;
  double dif2;
  int left;
  int right;
  double t1;
  double t2;
  double t3;
  double y1;
  double y2;
  double y3;

  if ( ndata < 3 )
  {
    cout << "\n";
    cout << "SPLINE_QUADRATIC_VAL - Fatal error!\n";
    cout << "  NDATA < 3.\n";
    exit ( 1 );
  }

  if ( ndata % 2 == 0 )
  {
    cout << "\n";
    cout << "SPLINE_QUADRATIC_VAL - Fatal error!\n";
    cout << "  NDATA must be odd.\n";
    exit ( 1 );
  }
//
//  Find the interval [ TDATA(LEFT), TDATA(RIGHT) ] that contains, or is
//  nearest to, TVAL.
//
  r8vec_bracket ( ndata, tdata, tval, &left, &right );
//
//  Force LEFT to be odd.
//
  if ( left % 2 == 0 )
  {
    left = left - 1;
  }
//
//  Copy out the three abscissas.
//
  t1 = tdata[left-1];
  t2 = tdata[left  ];
  t3 = tdata[left+1];

  if ( t2 <= t1 || t3 <= t2 )
  {
    cout << "\n";
    cout << "SPLINE_QUADRATIC_VAL - Fatal error!\n";
    cout << "  T2 <= T1 or T3 <= T2.\n";
    exit ( 1 );
  }
//
//  Construct and evaluate a parabolic interpolant for the data
//  in each dimension.
//
  y1 = ydata[left-1];
  y2 = ydata[left  ];
  y3 = ydata[left+1];

  dif1 = ( y2 - y1 ) / ( t2 - t1 );

  dif2 = ( ( y3 - y1 ) / ( t3 - t1 )
       - ( y2 - y1 ) / ( t2 - t1 ) ) / ( t3 - t2 );

  *yval = y1 + ( tval - t1 ) * ( dif1 + ( tval - t2 ) * dif2 );
  *ypval = dif1 + dif2 * ( 2.0 * tval - t1 - t2 );

  return;
}
//****************************************************************************80


/*
void timestamp ( )

// ****************************************************************************80
//
//  Purpose:
//
//    TIMESTAMP prints the current YMDHMS date as a time stamp.
//
//  Example:
//
//    31 May 2001 09:45:54 AM
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    08 July 2009
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    None
//
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct std::tm *tm_ptr;
  //size_t len;
  std::time_t now;

  now = std::time ( NULL );
  tm_ptr = std::localtime ( &now );

  len = std::strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm_ptr );

  std::cout << time_buffer << "\n";

  return;
# undef TIME_SIZE
}

*/
