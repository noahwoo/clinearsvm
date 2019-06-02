/*
 * linsvm_train.cpp
 *
 *  Created on: Mar 21, 2010
 *      Author: noah
 */

#include <fstream>
#include <string.h>
#include <sstream>
#include <stdlib.h>
#include "snglin.h"
#include "linsvm_train.h"
using namespace std;

void init_data(DATA* dat, int cap, int n) {
	dat->n = n;
	dat->m = 0;
	
	dat->ix = (int**)malloc(sizeof(int*)*cap);
	dat->x  = (float**)malloc(sizeof(float*)*cap);
	dat->lx = (int*)malloc(sizeof(int*)*cap);
	dat->y  = (int*)malloc(sizeof(int*)*cap);
	
	dat->capacity = cap;
}

void inc_data_capacity(DATA* dat, int inc) {
	
	dat->ix = (int**)realloc(dat->ix, sizeof(int*)*(inc+dat->capacity));
	dat->x  = (float**)realloc(dat->x, sizeof(float*)*(inc+dat->capacity));
	dat->lx = (int*)realloc(dat->lx, sizeof(int*)*(inc+dat->capacity));
	dat->y  = (int*)realloc(dat->y, sizeof(int*)*(inc+dat->capacity));
	dat->capacity += inc;
}

void free_data(DATA* dat) {
	
	for(int i=0;i<dat->capacity;++i) {
		if(dat->ix[i]) free(dat->ix[i]);
		if(dat->x[i]) free(dat->x[i]);
	}
	if(dat->ix) free(dat->ix);
	if(dat->x) free(dat->x);
	if(dat->lx) free(dat->lx);
	if(dat->y) free(dat->y);
}

void save_model(MODEL* mod, char* file) {
	FILE* fp = fopen(file,"w");
	if(!fp) {
		printf("Fail to write model to %s.\n",file);
		exit(-1);
	}
	fprintf(fp, "%d %d %lf %lf\n", mod->n, mod->nsv, mod->xi, mod->C);
	int nnz=0;
	for(int i=0; i<mod->n; ++i) {
		if(mod->weight[i] != 0) {
			nnz+=1;
		}
	}
	fprintf(fp,"%d", nnz);
	for(int i=0; i<mod->n; ++i) {
		if(mod->weight[i] != 0) {
			fprintf(fp," %d:%lf", i, mod->weight[i]);
		}
	}
	fprintf(fp, "\n");
	fclose(fp);
}

void read_model(MODEL* mod, char* file) {
	FILE* fp = fopen(file,"r");
	if(!fp) {
		printf("Fail to read model from %s.\n",file); exit(-1);
	}
	int ret;
	ret = fscanf(fp, "%d %d %lf %lf\n", &mod->n, &mod->nsv, &mod->xi, &mod->C);
	if(ret<=0) {
		printf("Fail to read model from %s.\n",file); exit(-1);
	}
	mod->weight = (double*) calloc(mod->n, sizeof(double));
	int nnz=-1, ind;
	double val;
	ret = fscanf(fp, "%d", &nnz);
	if(ret<=0) {
		printf("Fail to read model from %s.\n",file); exit(-1);
	}
	for(int i=0; i<nnz; ++i) {
		ret = fscanf(fp, " %d:%lf", &ind, &val);
		if(ret<=0) {
			printf("Fail to read model from %s.\n",file); exit(-1);
		}
		mod->weight[ind] = val;
	}
	fclose(fp);
}

int read_data(char* file, DATA* data, int tail_one) {
	// read ANSI data from file
	int nl, ll, dim;
	
	if( peek_data(file, &nl, &ll, &dim) < 0 ) {
		printf("Error when peek data. \n");
		return -1;
	}
	
	data->m  = nl;
	data->n  = dim;
	data->lx = (int*)malloc(sizeof(int) * nl);
	data->y  = (int*)malloc(sizeof(int) * nl);
	data->ix = (int**)malloc(sizeof(int*) * nl);
	data->x  = (float**)malloc(sizeof(float*) * nl);
	
	ifstream dat(file);
	int *ibuff   = (int*)malloc(sizeof(int) * (ll+2));
	float *xbuff = (float*)malloc(sizeof(float) * (ll+2));
	
	string line, item;
	int id, len, label, ind, prev_id;
	float val;

	
	ind = 0; prev_id = 0;

	while(getline(dat, line)) {
		// read empty or comment line
		if(line.empty() || line[0] == '#') {
			continue;
		}
		// format: <label>[ cost:<val>] <id>:<val>[ <id>:<val>]
		istringstream tokens(line);
		tokens >> item;
		if( sscanf(item.c_str(), "%d", &label) == 0 ) {
			printf("In line %d, error label format: %s", ind, item.c_str());
			exit(-1);
		}
		
		data->y[ind] = label;
		
		prev_id = 0;
		len = 0;
		while(tokens >> item) {
			if( sscanf(item.c_str(), "%d:%f", &id, &val) > 0 ) {
				if(prev_id > id) {
					printf("%d-th sample contains non-increase feature value pair.\n", ind);
					exit(-1);
				}
				ibuff[len] = id;
				xbuff[len] = val;
				
				prev_id = id;
				len += 1;
			} else {
				printf("Error format at line %d. \n", ind);
				exit(-1);
			}
		}
	
		data->lx[ind] = len;
		data->ix[ind] = (int*)malloc(sizeof(int) * len);
		data->x[ind]  = (float*)malloc(sizeof(float) * len);
		
		memcpy(data->ix[ind], ibuff, sizeof(int)*len);
		memcpy(data->x[ind], xbuff, sizeof(float)*len);
		
		ind += 1;
	}
	dat.close();
//#define DEBUG_READ	
#ifdef DEBUG_READ
	int i,j;
	printf("dat: m=%d, n=%d\n", data->m, data->n);
	for(i=0;i<data->m;++i){
		printf("%d", data->y[i]);
		for(j=0;j<data->lx[i];++j){
			printf(" %d:%f", data->ix[i][j], data->x[i][j]);
		}
		printf("\n");
	}
#endif
	free(ibuff);
	free(xbuff);
	return ind;
}

int peek_data(char* file, int* nl, int* ll, int* dim)
{
	(*nl) = 0; (*ll) = 0; (*dim) = 0;
	ifstream dat(file);
	if(!dat) {
		printf("Fail to open file %s.\n", file);
		return -1;
	}

	string line, item;

	(*nl)  = 0;
	(*ll)  = 0;
	(*dim) = 0;

	int id, len;
	float val;

	while(getline(dat, line)) {
		// read empty or comment line
		if(line.empty() || line[0] == '#') {
			continue;
		}
		// format: <label> <id>:<val>[ <id>:<val>]
		istringstream tokens(line);
		tokens >> item;

		len = 0;
		while(tokens >> item) {
			if( sscanf(item.c_str(), "%d:%f", &id, &val) > 0 ) {
				len += 1;
				if(id > *dim) {
					*dim = id;
				}
			} else {
				printf("Error format at line %d. \n", *nl);
				exit(-1);
			}
		}
		if(len > *ll) {
			*ll = len;
		}

		*nl += 1;
	}
	dat.close();
	(*dim) += 1;
	return 0;
}

double prod_ns(double* wgt, int n, int* ind, float* val, int len) {
	double sum = 0.0;
	for(int i=0; i < len; ++i) {
		if(ind[i]<n) {
			sum += wgt[ind[i]]*val[i];
		}
	}
	return sum;
}

void add_ns(double* wgt, int* ind, float* val, int len, double alpha) {
	for(int i=0; i < len; ++i) {
		wgt[ind[i]]+=val[i]*alpha;
	}
}

void add_to_fake(DATA* fake, double* wgt, DATA* dat) {
	if(fake->m >= fake->capacity) {
		inc_data_capacity(fake, fake->capacity*2);
	}
	int nnz=0;
	for(int i=0; i<dat->n; ++i) if(wgt[i]!=0) nnz+=1;
	if(nnz > 0) {
		fake->ix[fake->m] = (int*)malloc(sizeof(int)*nnz);
		fake->x[fake->m] = (float*)malloc(sizeof(float)*nnz);
		fake->lx[fake->m] = nnz;
		int ind=0;
		for(int i=0; i<dat->n; ++i) {
			if(wgt[i]!=0) {
				fake->ix[fake->m][ind] = i;
				fake->x[fake->m][ind] = wgt[i];
				ind+=1;
			}
		}
		fake->m += 1;
	}else{
		printf("Read full zero weight.\n");
		exit(-1);
	}
}

double prod_ss(DATA* dat, int ii, int jj) {
	int i=0,j=0;
	double sum=0.0;
	while(1) {
		if(i>=dat->lx[ii] || j>=dat->lx[jj]) break;
		if(dat->ix[ii][i] < dat->ix[jj][j]) {
			++i;
		}else if(dat->ix[ii][i] > dat->ix[jj][j]) {
			++j;
		}else{
			sum += dat->x[ii][i] * dat->x[jj][j]; 
			++i; ++j;
		}
	}
	return sum;
}

int linsvm_learn(DATA * dat, PARAM* parm, MODEL* mod) {
	
	// pre-setting the model
	mod->n = dat->n;
	mod->C = parm->C;
	mod->weight = (double*)calloc(mod->n, sizeof(double));
	// auxillary vector
	double* wgt = (double*)calloc(mod->n, sizeof(double));
	PGPARAM pgparm;
	DATA fake;
	QP qp;
	init_pgparam(&pgparm);
	init_data(&fake, parm->init_cap, dat->n);
	init_qp(&qp, 0, parm->init_cap);
	qp.C = parm->C;
	
	// main iteration
	double tol=-1.0, margin, resd;
	mod->xi = 0.0;
	int it=0, isteps, nsv;
	do {
		
		// solve begins from the 2nd iteration
		nsv = 0;
		if(it > 0) {
			isteps = solve(&qp, &pgparm);
			printf("#it=%d#: === solve subproblem in %d inner steps ===\n", it, isteps);
			// dump_qp(&qp);

			// update the weight with the solution
			memset(mod->weight, 0, sizeof(double)*mod->n);
			for(int i=0; i<fake.m; ++i) {
				if(qp.x[i] > 0) {
					add_ns(mod->weight, fake.ix[i], fake.x[i], fake.lx[i], qp.x[i]);
				}
			}

			mod->xi = 0.0;
			for(int i=0; i<fake.m; ++i) {
				margin = prod_ns(mod->weight, mod->n, fake.ix[i], fake.x[i], fake.lx[i]);
				resd = -qp.b[i]-margin;
				if(qp.x[i] > 0) {
					nsv += 1;
					if(resd > 0 && resd > mod->xi) {
						printf("#it=%d#: \\xi_c=%.4lf, margin=%.4lf, x[%d]=%.4lf\n", it, resd, margin, i, qp.x[i]);
						mod->xi = resd;
					}
				}
			}
		}
		// select the violating examples
		double rhs=0.0;
		if(it==0) {
			for(int i=0; i< dat->m; ++i) {
				add_ns(wgt, dat->ix[i], dat->x[i], dat->lx[i], dat->y[i]*1.0/dat->m);
				rhs += 1;
			}
		} else { 
			for(int i=0; i< dat->m; ++i) {
				margin = prod_ns(mod->weight, mod->n, dat->ix[i], dat->x[i], dat->lx[i]);
				if(dat->y[i] * margin < 1.0) {
					// x = x + \alpha * y
					add_ns(wgt, dat->ix[i], dat->x[i], dat->lx[i], dat->y[i]*1.0/dat->m);
					rhs += 1;
				}
			}
		}
		rhs /= dat->m;
		
		// construct the next QP problem
		add_to_fake(&fake, wgt, dat);
		printf("#it=%d#: rhs=%lf, #fake=%d, xi=%lf\n", it, rhs, fake.m, mod->xi);
		
		if(it > 0) { // check if iteration should terminate
			tol = 0.0;
			int i = fake.m-1;
			margin = prod_ns(mod->weight, mod->n, fake.ix[i], fake.x[i], fake.lx[i]);
			if(rhs - margin - mod->xi > tol) {
				tol = rhs - margin - mod->xi;
			}
			
			printf("#it=%d#: xi=%.4lf, tol=%.4lf, weight =", it, mod->xi, tol);
			for(int i=0; i<dat->n; ++i) {
				printf(" %.4lf", mod->weight[i]);
			}
			printf("\n");
			
			if(tol < parm->eps) {
				break;
			}
		}
		
		// construct the dual QP
		if(qp.capacity <= fake.m) {
			inc_qp_capacity(&qp, qp.capacity*2);
		}
		qp.n = fake.m;
		for(int i = 0; i<qp.n; ++i) {
			qp.Q[qp.n-1][i] = prod_ss(&fake, qp.n-1, i);
			qp.Q[i][qp.n-1] = qp.Q[qp.n-1][i]; 
		}
		qp.b[qp.n-1] = -rhs;
		qp.n += 1; // plus one for the trick
		
		it += 1;
		// for the next round
		memset(wgt, 0, sizeof(double)*mod->n);
		
	} while(it < parm->max_it);
	mod->nsv = nsv;
	// test train error:
	/* int corr=0;
	for(int i=0; i<dat->m; ++i) {
		margin = prod_ns(mod->weight, mod->n, dat->ix[i], dat->x[i], dat->lx[i]);
		if(dat->y[i]*margin>=0) {
			corr+=1;
		}
	}
	printf("Training Error: corr=%d,total=%d,accuracy=%.4lf\n", corr, dat->m, ((double)corr)/dat->m); */
	// release the space
	free(wgt);
	
	/* free_data(&fake);
	free_qp(&qp); */
	return it;
}
