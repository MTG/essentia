#include <string>

using std::string;

double basis_function_b_val ( double tdata[], double tval );
double basis_function_beta_val ( double beta1, double beta2, double tdata[],
  double tval );
double *basis_matrix_b_uni ( );
double *basis_matrix_beta_uni ( double beta1, double beta2 );
double *basis_matrix_bezier ( );
double *basis_matrix_hermite ( );
double *basis_matrix_overhauser_nonuni ( double alpha, double beta );
double *basis_matrix_overhauser_nul ( double alpha );
double *basis_matrix_overhauser_nur ( double beta );
double *basis_matrix_overhauser_uni ( void);
double *basis_matrix_overhauser_uni_l ( );
double *basis_matrix_overhauser_uni_r ( );
double basis_matrix_tmp ( int left, int n, double mbasis[], int ndata,
  double tdata[], double ydata[], double tval );
void bc_val ( int n, double t, double xcon[], double ycon[], double *xval,
  double *yval );
double bez_val ( int n, double x, double a, double b, double y[] );
double bp_approx ( int n, double a, double b, double ydata[], double xval );
double *bp01 ( int n, double x );
double *bpab ( int n, double a, double b, double x );
int chfev ( double x1, double x2, double f1, double f2, double d1, double d2,
  int ne, double xe[], double fe[], int next[] );
int d3_fs ( double a1[], double a2[], double a3[], int n, double b[], double x[] );
double *d3_mxv ( int n, double a[], double x[] );
double *d3_np_fs ( int n, double a[], double b[] );
void d3_print ( int n, double a[], string title );
void d3_print_some ( int n, double a[], int ilo, int jlo, int ihi, int jhi );
double *d3_uniform ( int n, int *seed );
void data_to_dif ( int ntab, double xtab[], double ytab[], double diftab[] );
double dif_val ( int ntab, double xtab[], double diftab[], double xval );
int i4_max ( int i1, int i2 );
int i4_min ( int i1, int i2 );
void least_set ( int point_num, double x[], double f[], double w[],
  int nterms, double b[], double c[], double d[] );
double least_val ( int nterms, double b[], double c[], double d[],
  double x );
void least_val2 ( int nterms, double b[], double c[], double d[], double x,
  double *px, double *pxp );
void least_set_old ( int ntab, double xtab[], double ytab[], int ndeg,
  double ptab[], double b[], double c[], double d[], double *eps, int *ierror );
double least_val_old ( double x, int ndeg, double b[], double c[], double d[] );
void parabola_val2 ( int ndim, int ndata, double tdata[], double ydata[],
  int left, double tval, double yval[] );
double pchst ( double arg1, double arg2 );
double r8_max ( double x, double y );
double r8_min ( double x, double y );
double r8_uniform_01 ( int *seed );
void r8vec_bracket ( int n, double x[], double xval, int *left, int *right );
void r8vec_bracket3 ( int n, double t[], double tval, int *left );
double *r8vec_even ( int n, double alo, double ahi );
double *r8vec_indicator ( int n );
void r8vec_order_type ( int n, double x[], int *order );
void r8vec_print ( int n, double a[], string title );
void r8vec_sort_bubble_a ( int n, double a[] );
double *r8vec_uniform ( int n, double b, double c, int *seed );
int r8vec_unique_count ( int n, double a[], double tol );
void r8vec_zero ( int n, double a[] );
int s_len_trim ( string s );
double spline_b_val ( int ndata, double tdata[], double ydata[], double tval );
double spline_beta_val ( double beta1, double beta2, int ndata, double tdata[],
  double ydata[], double tval );
double spline_constant_val ( int ndata, double tdata[], double ydata[], double tval );
double *spline_cubic_set ( int n, double t[], double y[], int ibcbeg, double ybcbeg,
  int ibcend, double ybcend );
double spline_cubic_val ( int n, double t[], double tval, double y[], double ypp[],
  double *ypval, double *yppval );
void spline_cubic_val2 ( int n, double t[], double tval, int *left, double y[],
  double ypp[], double *yval, double *ypval, double *yppval );
double *spline_hermite_set ( int ndata, double tdata[], double ydata[],
  double ypdata[] );
void spline_hermite_val ( int ndata, double tdata[], double c[], double tval,
  double *sval, double *spval );
double spline_linear_int ( int ndata, double tdata[], double ydata[], double a,
  double b );
void spline_linear_intset ( int int_n, double int_x[], double int_v[],
  double data_x[], double data_y[] );
void spline_linear_val ( int ndata, double tdata[], double ydata[],
  double tval, double *yval, double *ypval );
double spline_overhauser_nonuni_val ( int ndata, double tdata[],
  double ydata[], double tval );
double spline_overhauser_uni_val ( int ndata, double tdata[], double ydata[],
  double tval );
void spline_overhauser_val ( int ndim, int ndata, double tdata[], double ydata[],
  double tval, double yval[] );
void spline_pchip_set ( int n, double x[], double f[], double d[] );
void spline_pchip_val ( int n, double x[], double f[], double d[], int ne,
  double xe[], double fe[] );
void spline_quadratic_val ( int ndata, double tdata[], double ydata[],
  double tval, double *yval, double *ypval );
void timestamp ( );
