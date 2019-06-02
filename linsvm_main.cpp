/*
 * linsvm_main.cpp
 *
 *  Created on: Mar 21, 2010
 *      Author: noah
 */
#include <stdio.h>
#include <stdlib.h>
#include "snglin.h"
#include "linsvm_train.h"

void test_subsolver() {
	QP qp;
	init_qp(&qp,3,3);
	PGPARAM parm;
	init_pgparam(&parm);
	
	qp.C = 0.5;
	qp.Q[0][0] = 4; qp.Q[0][1] = 0;
	qp.Q[1][0] = 0; qp.Q[1][1] = 2;
	
	qp.b[0] = -1; qp.b[1] = -4;
	qp.x[0] = 0;  qp.x[1] = 0; 
	int it = solve(&qp, &parm);
	dump_qp(&qp);
	printf("solved in %d iterations.\n",it);	
}

int parse_argument(int argc, char* argv[], PARAM* param)
{
	int i;
	int ns = 0;
	// oo_svm -x v sample model
	for(i = 1; i < argc-1; ++i) {
		if(argv[i][0] == '-') {
			if(argv[i][2] == 0) {
				switch( argv[i][1] ) {
				case 'e':
					if(sscanf(argv[++i], "%lf", &param->eps)==0) {
						printf("Error tolerance: -e. \n");
						exit(1);
					}
					ns += 1;
					break;
				case 'c':
					if(sscanf(argv[++i], "%lf", &param->C) == 0) {
						printf("Error argument: -c. \n");
						exit(1);
					}
					ns += 1;
					break;
				case 'i':
					if(sscanf(argv[++i], "%d", &param->max_it) == 0) {
						printf("Error maximum iteration: -e. \n");
						exit(1);
					}
					ns += 1;
					break;
				case 'm':
					if(sscanf(argv[++i], "%d", &param->init_cap) == 0) {
						printf("Error capacity: -m. \n");
						exit(1);
					}
					ns += 1;
					break;
				default:
					printf("Unknown argument -%c. \n", argv[i][1]);
					exit(1);
					break;
				}
			}
		}
	}
	return ns;
}
void set_default_param(PARAM* parm) {
	parm->C        = 1.0;
	parm->eps      = 0.001;
	parm->init_cap = 20;
	parm->max_it   = 2000;
}

int main(int argc, char* argv[]) {
	printf("Hello World!\n");
	test_subsolver();
	printf("===============================\n");
	PARAM parm;
	DATA dat;
	
	int np, rp, nl, steps;
	char *datafile, *modelfile;

	set_default_param(&parm);
	
	np = parse_argument(argc, argv, &parm);

	rp = argc - 2*np - 1;
	if(rp == 2) {
		datafile  = argv[argc-2];
		modelfile = argv[argc-1];
	} else if(rp == 1) {
		datafile  = argv[argc-1];
		modelfile = "linsvm.mod";
	} else {
		printf("Incorrect arguments are specified for me, please check again.\n");
		exit(-1);
	}

	nl = read_data(datafile, &dat, 0);
	MODEL mod;
	steps = linsvm_learn(&dat, &parm, &mod);
	printf("Training complete in %d steps.\n", steps);
	save_model(&mod, modelfile);
	printf("Saving model to %s.\n", modelfile);
}
