/*
 * linsvm_classify_main.cpp
 *
 *  Created on: Mar 21, 2010
 *      Author: noah
 */
#include <stdio.h>
#include <stdlib.h>
#include "linsvm_train.h"

int main(int argc, char* argv[]) {
	printf("Hello World!\n");
	if(argc < 3) {
		printf("usage: %s data-file model-file [result-file]\n", argv[0]);
		exit(-1);
	}
	char *fdata = argv[1], *fmodel = argv[2];
	char *fresult;
	if(argc > 3) {
		fresult = argv[3];
	}else{
		fresult = "pred-result.txt";
	}
	
	MODEL mod;
	read_model(&mod, fmodel);
	
	// check model obj
	double obj=0.0;
	for(int i=0; i<mod.n; ++i) {
		obj += mod.weight[i] * mod.weight[i]; 
	}
	obj /= 2;
	obj += mod.C*mod.xi;
	printf("Read model complete: n=%d, xi=%lf, C=%lf, obj=%lf.\n", mod.n, mod.xi, mod.C, obj);
	
	DATA dat;
	int m = read_data(fdata, &dat, 0);
	printf("Read %d examples to predict.\n", m);
	
	int poscorr, negcorr, npos, nneg;
	poscorr = negcorr = npos = nneg = 0;
	double margin;
	
	FILE* fp = fopen(fresult, "w");
	if(!fp) {
		printf("Fail to write model to %s.\n", fresult);
		exit(-1);
	}
	
	double accu /*, prec, reca*/;
	for(int i = 0; i< dat.m; ++i) {
		margin = prod_ns(mod.weight, mod.n, dat.ix[i], dat.x[i], dat.lx[i]);
		fprintf(fp, "%d %lf\n", dat.y[i], dat.y[i]*margin);
		if(dat.y[i] > 0) {
			npos+=1;
			if(margin>0) {
				poscorr+=1;
			}
		}else if(dat.y[i] < 0) {
			nneg+=1;
			if(margin<0) {
				negcorr+=1;
			}
		}
	}

	accu = ((double)(poscorr+negcorr))/(npos+nneg);
	printf("Result: %d out of %d(%.4lf%%) are correctly classified.\n", 
			poscorr+negcorr, npos+nneg, 100*accu);
	fclose(fp);
}
