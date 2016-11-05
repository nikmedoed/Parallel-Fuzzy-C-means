/*
 * clustering.cpp
 *
 *  Created on: Oct 30, 2016
 *      Author: michael
 */

#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SPACE_SIZE 1000000.0

// Arrays. Preallocate on initialization
// Main
double *x; // Data points
double *c; // Cluster centroids
double *f; // Membership coefficients
// Auxiliary
double *sc; // Partial sums for cluster centroids
double *sdiv; // Partial sums for cluster centroid dividers
double *div_res; // Aggregation results for cluster centroid dividers
double *dist; // Distances to centroids
int *proceed; // Continuation flag
int *proceed_global; // Receptacle for global continuation status

// Singular variables
double eps; // Required precision
int d; // Number of data dimensions
double m; // Fuzzyness power
int lx; // Data point count
int lc; // Cluster count

bool iterate();

int init(int _lx, int _d, int _lc, double _m, double _eps, int p)
{
	if (_lx % p != 0)
		return 1;
	lx = _lx / p;
	d = _d;
	lc = _lc;
	m = _m;
	eps = _eps;
	x = (double *)malloc(lx * d * sizeof(double));
	c = (double *)malloc(lc * d * sizeof(double));
	f = (double *)malloc(lc * lx * sizeof(double));
	sc = (double *)malloc(d * lc * sizeof(double));
	sdiv = (double *)malloc(lc * sizeof(double));
	div_res = (double *)malloc(lc * sizeof(double));
	dist = (double *)malloc(lx * lc * sizeof(double));
	proceed = (int *)malloc(sizeof(int));
	proceed_global = (int *)malloc(sizeof(int));
	srand(time(0));
	for (int i = 0; i < lx; ++i)
		for (int j = 0; j < d; ++j)
			x[i*d + j] = rand() * SPACE_SIZE;
	for (int i = 0; i < lx; ++i)
	{
		double s = 0;
		for (int j = 0; j < lc; ++j)
		{
			f[i*d + j] = rand();
			s += f[i*d + j];
		}
		for (int j = 0; j < lc; ++j)
		{
			f[i*d + j] /= s;
		}
	}
	return 0;
}

int cluster(int iter)
{
	int iter_done = 0;
	*proceed_global = 1;
	while (iter_done < iter && *proceed_global != 0)
	{
		*proceed = (iterate()) ? 0 : 1;
		++iter_done;
		MPI_Allreduce(proceed, proceed_global, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	}
	return iter_done;
}

bool iterate()
{
	bool done = true;
	memset(sc, 0, d * lc * sizeof(double));
	memset(sdiv, 0, lc * sizeof(double));
	memset(div_res, 0, lc * sizeof(double));
	double mu;
	for (int i = 0; i < lx; ++i)
		for (int j = 0; j < lc; ++j)
		{
			mu = pow(f[i*lc + j], m);
			sdiv[j] += mu;
			for (int k = 0; k < d; ++k)
				sc[j*d + k] = mu * x[i*d + k];
		}
	MPI_Allreduce(sdiv, div_res, lc, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(sc, c, lc * d, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	for (int i = 0; i < lc; ++i)
		for (int j = 0; j < d; ++j)
			c[i*d + j] /= div_res[i];
	memset(dist, 0, lx * lc * sizeof(double));
	for (int i = 0; i < lx; ++i)
		for (int j = 0; j < lc; ++j)
			for (int k = 0; k < d; ++k)
				dist[i*lc + j] += (x[i*d + k] - c[j*d + k]) * (x[i*d + k] - c[j*d + k]);
	for (int i = 0; i < lx; ++i)
	{
		for (int j = 0; j < lc; ++j)
		{
			double f_prev = f[i*lc + j];
			f[i*lc + j] = 0;
			for (int k = 0; k < lc; ++k)
				f[i*lc + j] += dist[i*d + j] / dist[i*d + k];
			f[i*lc + j] = 1.0 / pow(f[i*lc + j], 1.0 / (m - 1));
			if (done && fabs(f_prev - f[i*lc + j]) > eps)
				done = false;
		}
	}

	return done;
}

int terminate()
{
	free(x);
	free(c);
	free(f);
	free(sc);
	free(sdiv);
	free(div_res);
	free(dist);
	free(proceed);
	free(proceed_global);
	return 0;
}
