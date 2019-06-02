/*
 * snglin.c
 *
 *  Created on: Mar 21, 2010
 *      Author: noah
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "snglin.h"

int compare(const void* a, const void* b) {
	if(*(double*)(a) > *(double*)(b)) {
		return 1;
	}else if(*(double*)(a) < *(double*)(b)) {
		return -1;
	}else{
		return 0;
	}
}

void init_qp(QP* qp, int dim, int cap) {
	int i;
	qp->n = dim;
	qp->C = 1.0;
	qp->capacity = cap;
	/* allocate space */
	qp->Q = (double**)malloc(sizeof(double*) * cap);
	for(i=0;i<cap;++i) {
		qp->Q[i] = (double*)malloc(sizeof(double) * cap);
	}
	qp->b = (double*)malloc(sizeof(double) * cap);
	qp->x = (double*)malloc(sizeof(double) * cap);
}

void inc_qp_capacity(QP* qp, int inc) {
	
	qp->b = (double*)realloc(qp->b, sizeof(double) * (qp->capacity+inc));
	qp->x = (double*)realloc(qp->x, sizeof(double) * (qp->capacity+inc));
	
	qp->Q = (double**)realloc(qp->Q, sizeof(double*) * (qp->capacity+inc));
	for(int i=0;i<qp->n;++i) {
		qp->Q[i] = (double*)realloc(qp->Q[i], sizeof(double) * (qp->capacity+inc));
	}
	// create new for the additional array
	for(int i = qp->n; i< (qp->capacity+inc); ++i) {
		qp->Q[i] = (double*)malloc(sizeof(double) * (qp->capacity+inc));
	}
	
	qp->capacity += inc;
}

void free_qp(QP* qp) {
	int i;
	if(qp->x) free(qp->x);
	if(qp->b) free(qp->b);
	if(qp->Q) {
		for(i=0;i<qp->n;++i) {
			if(qp->Q[i]) free(qp->Q[i]);
		}
		free(qp->Q);
	}
}

void init_pgparam(PGPARAM* parm) {
	parm->MAX_ITER = 5000;
	parm->eps = 0.00001;
	parm->alpha_min = 1E-10;
	parm->alpha_max = 1E10;
}

void dump_qp(QP* qp) {
	
	printf("min 1/2 x'Qx+b'x \n");
	printf(" s.t. 0<= \\sum_i x_i <= C\n");
	printf("dim=%d\n",qp->n);
	printf("Q=\n");

	double obj=0.0;
	for(int i=0;i<qp->n-1;++i) {
		for(int j=0;j<qp->n-1;++j) {
			printf(" %lf",qp->Q[i][j]);
			obj += qp->Q[i][j]*qp->x[i]*qp->x[j];
		}
		printf("\n");
	}
	printf("b =");
	obj /= 2;
	for(int i=0;i<qp->n-1;++i) {
		printf(" %lf",qp->b[i]);
		obj += qp->b[i]*qp->x[i];
	}
	printf("\n");
	printf("c = %lf\n",qp->C);
	// optimal solution
	printf("optimal x =");
	for(int i=0;i<qp->n-1;++i) {
		printf(" %lf",qp->x[i]);
	}
	printf("\n");
	
	printf("obj = %lf\n",obj);
}

void project(QP* qp, double* gx, double* x) {
	int i,t,n;
	double sx, lambda;
	double *aux;
	sx = qp->C;
	n = qp->n;
	aux = (double*)malloc(sizeof(double)*n);
	for(i=0;i<n;++i) {
		aux[i] = -gx[i];
		sx += aux[i];
	}

	lambda = 0.0;
	/* Arrays.sort(aux); */
	qsort(aux, n, sizeof(double), compare);
// #define DEBUG_PROJ
#ifdef DEBUG_PROJ
	for(i = 0; i<n; ++i) {
		printf("aux[%d]=%lf ",i,aux[i]);
	}
	printf("\n"); */
#endif
	if(sx/n >= aux[n-1]) {
		lambda = sx/n;
		t=n;
	}else{
		for(t=n-1; t>0; --t) {
			sx -= aux[t];
			if(sx/t >= aux[t-1]) {
				lambda = sx/t;
				break;
			}
		}
	}

	sx = 0.0;
	for(i=0;i<n;++i) {
		x[i] = gx[i]+lambda;
		if(x[i]<0.0) {
			x[i] = 0.0;
		}
		sx += x[i];
	}	
	free(aux);
}

void gradient(QP* qp, double* xbar, double* grad) {
	int i,n;
	n = qp->n;
	for(i=0;i<n-1;++i) {
		grad[i] = 0.0;
		for(int j=0;j<n-1;++j) {
			grad[i] += qp->Q[i][j]*xbar[j];
		}
		grad[i] += qp->b[i];
	}
	grad[n-1] = 0.0;
}

void Qx(QP* qp, double *xbar, double* x) {
	int n;
	n = qp->n;
	// Auxiliary function
	for(int i=0;i<n-1;++i) {
		x[i] = 0.0;
		for(int j=0;j<n-1;++j) {
			x[i] += qp->Q[i][j]*xbar[j];
		}
	}
	x[n-1] = 0.0;
}

double tolerance(QP* qp, double *g, double *pg, double *x) {
	int i,n;
	double viol;
	viol = 0.0;
	n = qp->n;
	for(i=0;i<n;++i) {
		if(fabs(pg[i]) > viol) {
			viol = fabs(pg[i]);
		}
	}
	return viol;
}

int solve(QP* qp, PGPARAM* param) {
	int i, iter;
	double gd, max, ak, bk, akold, bkold, lamnew, alpha;

	/*** variables for the adaptive nonmonotone linesearch ***/
	int    L, llast;
	double fr, fbest, fv, fc, fv0, tol;
	int n = qp->n;
	/*** arrays allocation ***/
	double* g  = (double*)malloc(sizeof(double)*n);
	double* y  = (double*)malloc(sizeof(double)*n);
	double* tempv = (double*)malloc(sizeof(double)*n);
	double* d  = (double*)malloc(sizeof(double)*n);
	double* Ad = (double*)malloc(sizeof(double)*n);
	double* t  = (double*)malloc(sizeof(double)*n);
	double* xplus = (double*)malloc(sizeof(double)*n);
	double* tplus = (double*)malloc(sizeof(double)*n);
	double* sk = (double*)malloc(sizeof(double)*n);
	double* yk = (double*)malloc(sizeof(double)*n);

	for (i = 0; i < n; ++i) {
		tempv[i] = qp->x[i];
	}

	project(qp, tempv, qp->x);

	gradient(qp, qp->x, g);
	for (i = 0; i < n; ++i) {
		t[i] = g[i]-qp->b[i];
	}

	for(i = 0; i < n; ++i) {
		y[i] = qp->x[i]-g[i];
	}
	project(qp, y,tempv);

	max = param->alpha_min;
	for(i = 0; i < n; ++i) {
		y[i] = tempv[i] - qp->x[i];
		if(fabs(y[i])>max) {
			max = fabs(y[i]);
		}
	}
	iter = 0;
	// stop directly
	if(max < param->eps*1E-4) {
		return iter;
	}
	// initial BB step-length
	alpha = 1.0/max;
	// initial objective
	fv0   = 0.0;
	for (i = 0; i < n; i++) {
		fv0 += qp->x[i]*(0.5*t[i]+qp->b[i]);
	}

	/*** adaptive nonmonotone linesearch ***/
	L     = 2;
	fr    = param->alpha_max;
	fbest = fv0;
	fc    = fv0;
	llast = 0;
	akold = bkold = 0.0;

	tol = 0.0;
	// main project gradient iteration
	for (iter = 1; iter <= param->MAX_ITER; iter++) {
		// move along gradient direction with BB step
		for (i = 0; i < n; i++) {
			tempv[i] = qp->x[i] - alpha*g[i];
		}
		// project back the trial point tempv to y
		project(qp, tempv, y);

		gd = 0.0;
		for (i = 0; i < n; i++) {
			d[i] = y[i] - qp->x[i];
			gd  += d[i] * g[i];
		}

		Qx(qp, d, Ad);

		ak = 0.0;
		for (i = 0; i < n; i++) {
			ak += d[i] * d[i];
		}

		bk = 0.0;
		for (i = 0; i < n; i++)
			bk += d[i]*Ad[i];

		// trial step-length
		if (bk > param->eps*ak && gd < 0.0)    // ak is normd
			lamnew = -gd/bk;
		else
			lamnew = 1.0;

		fv = 0.0;
		for (i = 0; i < n; i++)
		{
			xplus[i] = qp->x[i] + d[i];
			tplus[i] = t[i] + Ad[i];
			fv      += xplus[i] * (0.5*tplus[i] + qp->b[i]);
		}

		// move forward with step size lamnew
		if ((iter == 1 && fv >= fv0) || (iter > 1 && fv >= fr))
		{
			fv = 0.0;
			for (i = 0; i < n; i++)
			{
				xplus[i] = qp->x[i] + lamnew*d[i];
				tplus[i] = t[i] + lamnew*Ad[i];
				fv      += xplus[i] * (0.5*tplus[i] + qp->b[i]);
			}
		}

		// update gradient and x
		for (i = 0; i < n; i++)
		{
			sk[i] = xplus[i] - qp->x[i];
			yk[i] = tplus[i] - t[i];
			qp->x[i]  = xplus[i];
			t[i]  = tplus[i];
			g[i]  = t[i] + qp->b[i];
		}

		// update the line search control parameters

		if (fv < fbest)
		{
			fbest = fv;
			fc    = fv;
			llast = 0;
		}
		else
		{
			fc = (fc > fv ? fc : fv);
			llast++;
			if (llast == L)
			{
				fr    = fc;
				fc    = fv;
				llast = 0;
			}
		}

		ak = bk = 0.0;
		for (i = 0; i < n; i++)
		{
			ak += sk[i] * sk[i];
			bk += sk[i] * yk[i];
		}

		if (bk <= param->eps*ak || bk <= param->eps)
			alpha = param->alpha_max;
		else
		{
			if (bkold <= param->eps*akold)
				alpha = ak/bk;
			else
				alpha = (akold+ak)/(bkold+bk);

			if (alpha > param->alpha_max)
				alpha = param->alpha_max;
			else if (alpha < param->alpha_min)
				alpha = param->alpha_min;
		}

		akold = ak;
		bkold = bk;

		tol = tolerance(qp, g, sk, qp->x);
		if(tol < param->eps) {
			break;
		}
	}

	// free space
	free(g);
	free(y);
	free(tempv);
	free(d);
	free(Ad);
	free(t);
	free(xplus);
	free(tplus);
	free(sk);
	free(yk);
	
	return iter;	
}
