/*
 * driver.cpp
 *
 *  Created on: Oct 31, 2016
 *      Authors: michael & nikmedoed
 */

// for build
// mpixlcxx clustering.cpp driver.cpp -o fcm     
// for runnig on IBM Blue Gene\P on MSU CMC
// mpisubmit.bg -w 00:05:00 -m SMP -n 512 fcm -- 4 10 2 0.000001 1024 1048576 2 10 4
 
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int init(int _lx, int _d, int _lc, double _m, double _eps, int p);
int cluster(int iter);
int terminate();

int main(int argc, char **argv)
{
	int er;
	if ((er = MPI_Init(&argc,&argv)) != MPI_SUCCESS)
	{
		printf("Launch error, program terminated");
		MPI_Abort(MPI_COMM_WORLD, er);
	}
	int rnk;
	MPI_Comm_rank(MPI_COMM_WORLD,&rnk);
	int p;
	MPI_Comm_size(MPI_COMM_WORLD,&p);

	int exp = strtol(argv[1], NULL, 10);
	int d = strtol(argv[2], NULL, 10);
	double m = strtod(argv[3], NULL);
	double eps = strtod(argv[4], NULL);
	int lx_min = strtol(argv[5], NULL, 10);
	int lx_max = strtol(argv[6], NULL, 10);
	int lx_q = strtol(argv[7], NULL, 10);
	int lc = strtol(argv[8], NULL, 10);
	int iter = strtol(argv[9], NULL, 10);

	setbuf(stdout, NULL);

	if (rnk == 0)
	{
		printf("%d, %d, %f, %f, %d, %d, %d, %d, %d\n", exp, d, m, eps, lx_min, lx_max, lx_q, lc, iter);
		printf("%d\n", p);
	}

	for (int lx = lx_min; lx <= lx_max; lx *= lx_q)
	{
		double t = 0;
		int s_iter = 0;
		for (int i = 0; i < exp; ++i)
		{
			init(lx, d, lc, m, eps, p);
			MPI_Barrier(MPI_COMM_WORLD);
			t -= MPI_Wtime();
			s_iter += cluster(iter);
			t += MPI_Wtime();
			terminate();
		}
		t /= s_iter;
		if (rnk == 0)
			printf("%f ", t);
	}

	MPI_Finalize();
}
