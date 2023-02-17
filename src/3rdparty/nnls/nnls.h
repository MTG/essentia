#ifndef NNLS_H
#define NNLS_H

#ifdef __cplusplus
extern "C" {
#endif

int nnls(float *a, int mda, int m, int n, 
	 			 float *b, float *x, float *rnorm, 
	 			 float *w, float *zz, int *index,
				 int *mode);

int g1(float* a, float* b, float* cterm, 
			 float* sterm, float* sig);

float d_sign(float a, float b);

int h12(int mode, int* lpivot, int* l1,
        int m, float* u, int* iue, float* up,
				float* c__, int* ice, int* icv,
				int* ncv);

#ifdef __cplusplus
}
#endif

#endif

