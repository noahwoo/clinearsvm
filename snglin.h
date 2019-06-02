/*
 * snglin.h
 *
 *  Created on: Mar 21, 2010
 *      Author: noah
 */

#ifndef SNGLIN_H_
#define SNGLIN_H_

typedef struct _PGParam {
	/* termination */
	int MAX_ITER;
	double eps;
	/* BB Step bound */
	double alpha_min;
	double alpha_max;
} PGPARAM;


typedef struct _QP {
	int n;
	double **Q;
	double *b;
	double *x;
	
	double C; 
	
	int capacity;
} QP;

void init_qp(QP* qp, int dim, int cap);
void inc_qp_capacity(QP* qp, int inc);
void free_qp(QP* qp);

int compare(void* a, void* b);
void init_pgparam(PGPARAM* parm);
void project(QP* qp, double* gx, double* x);
void gradient(QP* qp, double* xbar, double* grad);
double tolerance(double *g, double *pg, double *x);
void Qx(QP* qp, double *xbar, double* x);
void dump_qp(QP* qp);

int solve(QP* qp, PGPARAM* param);

#endif /* SNGLIN_H_ */
