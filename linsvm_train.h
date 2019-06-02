/*
 * linsvm_train.h
 *
 *  Created on: Mar 21, 2010
 *      Author: noah
 */

#ifndef LINSVM_TRAIN_H_
#define LINSVM_TRAIN_H_

typedef struct _data {
	int *y;
	int **ix;
	float **x;
	int *lx;
	int m; // number of document
	int n; // number of feature
	int capacity; // capacity
} DATA;

typedef struct _param {
	double eps;
	double C;
	int max_it;
	int init_cap;
} PARAM;

typedef struct _model {
	/* data info */
	int n;
	int nsv;
	double C;
	double xi;
	double* weight;
} MODEL;

int read_data(char* file, DATA* data, int tail_one);
int peek_data(char* file, int* nl, int* ll, int* dim);
void init_data(DATA* dat, int m, int n);
void inc_data_capacity(DATA* dat, int inc);
int linsvm_learn(DATA * dat, PARAM* parm, MODEL* mod);
double prod_ss(DATA* dat, int ii, int jj);
void add_to_fake(DATA* fake, double* wgt, DATA* dat);
void add_ns(double* wgt, int* ind, float* val, int len, double alpha);
double prod_ns(double* wgt, int n, int* ind, float* val, int len);
void save_model(MODEL* mod, char* file);
void read_model(MODEL* mod, char* file);
#endif /* LINSVM_TRAIN_H_ */
