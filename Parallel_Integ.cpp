#define _USE_MATH_DEFINES // for C++
#define FFTW_ESTIMATE (1U << 6)
#include <math.h>
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <omp.h>
#include "fftw3.h"	//instead of this include the file fftw_threads.h
#include <valarray> // interesting stuff
#include <iostream> //for cout..
#include <vector>
#include <omp.h>
#include <cstring>

static const int nx = 256000; // 1000; //256;
static const int ny = 256;	  // 1000; //256;
static const int nyk = ny / 2 + 1;
static const int ncomp = 2;
using namespace std;
//#define size 3
// g++ U.cpp -lfftw3_threads -lfftw3 -lm -g -fopenmp-o ./Parallel/test.out && valgrind --track-origins=yes ./Parallel/test.out
//  g++ U.cpp -lfftw3_threads -lfftw3 -lm -g -fopenmp -o ./Parallel/test.out
// g++ U.cpp -lfftw3_threads -lfftw3 -lm -g -fopenmp -o test.out
// g++ U.cpp -lfftw3_threads -lfftw3 -lm -g -o ./Parallel/test.out
// g++ U.cpp -lfftw3 -lm -g -fopenmp -o test.out
// g++ U.cpp -lfftw3_threads -lfftw3 -lm -g -o ./Parallel/test.out && valgrind --track-origins=yes ./Parallel/test.out
int main();
void r2cfft(double rArr[], double cArr[][ncomp]);
void print2DPhysArray(double arr[]);
double makeSpatialMesh2D(double dx, double dy, double xarr[], double yarr[]);
void Arr3DArr2DMult(double arr3D[][ncomp], double arr2D[], double arrOut[][ncomp]);
void iArr3DMult(double arr[][ncomp], double arrOut[][ncomp]);
void derivk(double vark[][ncomp], double k[], double derivative[][ncomp]);
void makeFourierMesh2D(double Lx, double Ly, double KX[], double KY[], double ksqu[], double ninvksqu[], int FourierMeshType);
double mag1DArray(double arr[]);
void calcCollFreqk(double nek[][ncomp], double Tik[][ncomp], double Tek[][ncomp], double kb, double eps0, double mi, double me, double ri, double rn, double nn, double Oci, double Oce, double e, double nuink[][ncomp], double nuiek[][ncomp], double nuiik[][ncomp], double nuenk[][ncomp], double nueek[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double invnk[][ncomp]);
void c2rfft(double cArr[][ncomp], double rArr[]);
void Arr2DArr2DMult(double arrIn0[], double arrIn1[], double arrOut[]);
void convolve2D(double fk[][ncomp], double gk[][ncomp], double fgk[][ncomp]);
void rscalArr3DMult(double arr[][ncomp], double rscal, double arrOut[][ncomp]);
void laplaciank(double vark[][ncomp], double ksqu[], double derivative[][ncomp]);
void calcPotSourcek(double dndxk[][ncomp], double dndyk[][ncomp], double Pik[][ncomp], double Pek[][ncomp], double nuink[][ncomp], double nuiek[][ncomp], double nuenk[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double Oci, double Oce, double u[], double B[], double ksqu[], double potSourcek[][ncomp]);
void absComp(double arr3D[][ncomp], double arrOut[]);
double max2Dk(double arr2D[]);
double max2D(double arr2D[]);
double max_absComp(double arr3D[][ncomp]);
int potentialk(double invnk[][ncomp], double dndxk[][ncomp], double dndyk[][ncomp], double phik[][ncomp], double potSourcek[][ncomp], double kx[], double ky[], double ninvksqu[], double err_max, int max_iter);
void print2DArrf(char filename[], double arr[]);
void calcV_ExBk(double dphidxk[][ncomp], double dphidyk[][ncomp], double B[], double B2, double vexbkx[][ncomp], double vexbky[][ncomp]);
void Arr2DArr2DDiv(double arrIn0[], double arrIn1[], double arrOut[]);
void fourierDivision2D(double fk[][ncomp], double gk[][ncomp], double fgk[][ncomp]);
void calc_diamag(double dpdxk[][ncomp], double dpdyk[][ncomp], double B[], double B2, double qa, double nak[][ncomp], double diamagxk[][ncomp], double diamagyk[][ncomp]);
double Arr1DMax(double arr[], int arrLength);
double calc_dt(double U[], double vexbx[], double vexby[], double diamagxi[], double diamagyi[], double diamagxe[], double diamagye[], double cfl, double kmax, double maxdt);
void Arr3DArr3DAdd(double arr0[][ncomp], double arr1[][ncomp], double arrOut[][ncomp]);
void calc_residualn(double vexbxk[][ncomp], double vexbyk[][ncomp], double nink[][ncomp], double residnoutk[][ncomp], double kx[], double ky[]);
void calc_residualt(double voxk[][ncomp], double voyk[][ncomp], double tempink[][ncomp], double tempoutk[][ncomp], double kx[], double ky[]);
void calc_sourcen(double ksqu[], double nk[][ncomp], double d, double sourcenk[][ncomp]);
void RK4(double f[][ncomp], double dt, double residual[][ncomp], double source[][ncomp], int stage, double fout[][ncomp]);
void print2DB(char filename[], double arr[]);

// g++ Parallel_Inetg.cpp -Wall -lfftw3 -lm -o3 -pg -o ./test_gprof
// Try with optimization option for faster multiplication?
// g++ Parallel_Inetg.cpp -lfftw3 -lm -pg -o ./test_gproff

// ./test_gprof
//  gprof test_gprof gmon.out > profile-data1.txt

int main(void)
{
	// printf("\n main() starts...\n");

	double saveFrequency = 1.0; // Save data every this many time steps
	double dt_max = 0.1;		// Set max allowable time step
	double tend = 1000.;		// 1000.     // Set the end time
	double err_max = 1.e-8;		// Set max allowable error for potential solver
	double CFL = 3.;			// Set CFL number. We can just leave this at 3.
	double Dart = 0;			// Set artifical diffusion constants
	// change Dart to 1e3 and 7e3
	// Try 7e5
	// modify D artif 1e3: different than GDI TGI, func off grid size, tot box size
	int phi_iter_max = 500; // Max allowable iterations for potential solver

	int saveNum = 1;

	// Calculated parameters
	int iter_max = tend / dt_max;
	// int fftw_threads_init(void); // should only be called once and (in your main()function), performs any one-time initialization required to use threads on your system. It returns zeros if succesful and a non-zero if not (error)
	//  hyperbolic tan paramters
	double L = 2 * M_PI;			  // useless
	double Lx = 200000., Ly = 80000.; // different
	// double Lx = L, Ly = L;
	double dx = Lx / nx;
	double dy = Ly / ny;
	// I.Cs:  Parameters for initializing hyperbolic tangent IC
	double xg = Lx * (19. / 24); // xg center of hyperbolic tangent func
	double m = 0.5;
	double lg = 12000.; // lg is width of hyperbolic tangent func
	double o = 1.;
	double a = (o - m) / 2.; // difference from background?
	double c = -xg;
	double d = (o + m) / 2.; // size of the cloud ??
	double a2 = -1 * a;
	double c2 = -Lx - c;
	double d2 = d - m;

	double b = abs(((((tanh(log(m / o) / 4)) * a) + d) * (cosh(log(m / o) / 4)) * (cosh(log(m / o) / 4))) / (a * lg));
	// b = abs(((((tanh(log(m/o)/4)) * a) + d) * pow(cosh(log(m/o)/4), 2))/(a * lg));
	double bg = 2. / lg / lg;

	// Set physical constants
	double e = 1.602E-19;
	double kb = 1.38E-23;
	double me = 9.109E-31;
	double eps0 = 8.854E-12;
	double nn = 1.E14;			 // Neutral particle number density
	double mi = 16. * 1.672E-27; // Mass O+
	double mn = mi;				 // Mass O
	double ri = 152.E-12;		 // Effective collision radius
	double rn = ri;
	// Set magnetic field. Also calculate different properties of it
	double B[3] = {0, 0, 5E-5};
	double Bmag = mag1DArray(B);
	double B2 = Bmag * Bmag;

	// Set neutral wind. Generally, the last element will be 0.
	double u[3] = {-500, 0, 0};

	double *ne;
	ne = (double *)fftw_malloc(nx * ny * sizeof(double));
	// test std instead
	// std::vector<double*> ne(nx*ny); //init in a for loop

	fftw_complex *nek;
	nek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(nek, 42, nx * nyk * sizeof(fftw_complex));
	// test std using complex instead
	double *XX;
	XX = (double *)fftw_malloc(nx * ny * sizeof(double));
	// print2DPhysArray(XX);

	double *YY;
	YY = (double *)fftw_malloc(nx * ny * sizeof(double));

	double kmax = makeSpatialMesh2D(dx, dy, XX, YY); // returns XX, YY, ..

	double *Ti;
	Ti = (double *)fftw_malloc(nx * ny * sizeof(double));

	fftw_complex *Tik;
	Tik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Tik, 42, nx * nyk * sizeof(fftw_complex));

	double *Te;
	Te = (double *)fftw_malloc(nx * ny * sizeof(double));

	fftw_complex *Tek;
	Tek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Tek, 42, nx * nyk * sizeof(fftw_complex)); // test extra to testbed

	double *Pi;
	Pi = (double *)fftw_malloc(nx * ny * sizeof(double));

	fftw_complex *Pik;
	Pik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Pik, 42, nx * nyk * sizeof(fftw_complex)); // test

	double *Pe;
	Pe = (double *)fftw_malloc(nx * ny * sizeof(double));

	fftw_complex *Pek;
	Pek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Pek, 42, nx * nyk * sizeof(fftw_complex)); // test

	double *phi;
	phi = (double *)fftw_malloc(nx * ny * sizeof(double));

	fftw_complex *phik;
	phik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(phik, 42, nx * nyk * sizeof(fftw_complex)); // test

	double *kx;
	kx = (double *)fftw_malloc(nx * nyk * sizeof(double));
	memset(kx, 42, nx * nyk * sizeof(double)); // test extra to testbed
	double *ky;
	ky = (double *)fftw_malloc(nx * nyk * sizeof(double));
	memset(ky, 42, nx * nyk * sizeof(double)); // test extra to testbed

	double *ksqu;
	ksqu = (double *)fftw_malloc(nx * nyk * sizeof(double));
	memset(ksqu, 42, nx * nyk * sizeof(double)); // test

	double *ninvksqu;
	ninvksqu = (double *)fftw_malloc(nx * nyk * sizeof(double));
	memset(ninvksqu, 42, nx * nyk * sizeof(double)); // test extra to testbed

	// Make the Fourier grid
	int FourierMeshType = 1; // 1;// 1; // this "flag" should not be double, this should generally be avoided so make it int and compare it against ints instead
	// Make the Fourier grid
	makeFourierMesh2D(Lx, Ly, kx, ky, ksqu, ninvksqu, FourierMeshType);
	// Residuals and source terms
	// print2DPhysArray(kx);

	fftw_complex *sourcetk;
	sourcetk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// 	#pragma omp parallel for num_threads(8) collapse(3)
	clock_t start_time = clock();
	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				sourcetk[j + nyk * i][k] = j + nyk * i; // 0;
				// printf("i = %d, j= %d, threadId = %d \n", i, j, omp_get_thread_num());
			}
		}
	}
	clock_t end = clock();
	cout << "Time in s: " << (double)(end - start_time) / CLOCKS_PER_SEC << "s\n";
	// print2DPhysArray(sourcetk);

	// fftw_complex *nek_old;
	// nek_old = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));

	// fftw_complex *Tik_old;
	// Tik_old = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));

	// fftw_complex *Tek_old;
	// Tek_old = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));

	// Initialize velocities

	// start a threading loop
	//   for (int nThreads= 1; nThreads<=16; nThreads++){
	//	int nThreads = 8;
	//	omp_set_num_threads(nThreads); // asks openmp to create a team of threads
	//		omp_get_max_threads();
	//   clock_t start_time = clock(); // or use: start_time= omp_get_wtime();
	//#pragma omp parallel for // start parallel region or fork threads
	//#pragma omp single // cause one thread to print number of threads used by the program
	//      printf("num_threads = %d", omp_get_num_threads());
	//#pragma omp for reduction(+ : sum) // this is to split up loop iterations among the team threads. It include 2 clauses:1) creates a private variable and 2) cause threads to compute their sums locally and then combine their local sums to a single gloabl value
	//#pragma omp for // more natural option to parallel execution of for loops: no need for new loop bounds
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			// tem[j + ny*i] =  tanh(b*(XX[j + ny*i] + c));
			// tem2[j + ny*i] = .02*cos(4*M_PI * YY[j + ny*i]/Ly);
			// ne[j + ny*i] =  (a * tanh(b*(XX[j + ny*i] + c)) + d + a2 *tanh(b*(XX[j + ny*i] + c))+ d2 + .02*cos(4*M_PI * YY[j + ny*i]/Ly) *( exp( - bg * ((XX[j + ny*i] - xg)*(XX[j + ny*i] - xg))) + exp(-bg* ( XX[j + ny*i] - Lx + xg)*( XX[j + ny*i] - Lx + xg))  ) ) * 1.E11;
			ne[j + ny * i] = (a * tanh(b * (XX[j + ny * i] + c)) + d + a2 * tanh(b * (XX[j + ny * i] + c2)) + d2 + .02 * cos(4 * M_PI * YY[j + ny * i] / Ly) * (exp(-bg * pow(XX[j + ny * i] - xg, 2)) + exp(-bg * pow(XX[j + ny * i] - Lx + xg, 2)))) * 1.E11;
			phi[j + ny * i] = 1.;

			Ti[j + ny * i] = 1000.;
			Te[j + ny * i] = 1000.;

			// Pi[j + ny*i] = ne[j + ny*i] * 1000. * 1.38E-23; // test set 1000 instead of Te
			Pi[j + ny * i] = ne[j + ny * i] * Ti[j + ny * i] * kb;
			// Pe[j + ny*i] = ne[j + ny*i] *1000.* 1.38E-23;
			Pe[j + ny * i] = ne[j + ny * i] * Te[j + ny * i] * kb;
		}
	}

	// convert to Fourier space
	r2cfft(ne, nek);
	r2cfft(Ti, Tik);
	r2cfft(Te, Tek);
	r2cfft(phi, phik);
	r2cfft(Pi, Pik);
	r2cfft(Pe, Pek);

	// print2DPhysArray(ne);

	fftw_complex *dndxk;
	dndxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dndxk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *dndyk;
	dndyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dndyk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *dphidxk;
	dphidxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dphidxk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *dphidyk;
	dphidyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dphidyk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *dpedxk;
	dpedxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpedxk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *dpedyk;
	dpedyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpedyk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *dpidxk;
	dpidxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpidxk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *dpidyk;
	dpidyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpidyk, 42, nx * nyk * sizeof(fftw_complex)); // test

	derivk(nek, kx, dndxk);
	derivk(nek, ky, dndyk);

	derivk(phik, kx, dphidxk);
	derivk(phik, ky, dphidyk);

	derivk(Pek, kx, dpedxk);
	derivk(Pek, ky, dpedyk);
	derivk(Pik, kx, dpidxk);
	derivk(Pik, ky, dpidyk);

	// Calculate ion nd electron gyrofrequencies - qb/m

	double Oci = e * Bmag / mi;
	double Oce = e * Bmag / me;

	// Initialize and calculate collision frequencies

	fftw_complex *nuink;
	nuink = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(nuink, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *nuiek;
	nuiek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(nuiek, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *nuiik;
	nuiik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(nuiik, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *nuenk;
	nuenk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(nuenk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *nueek;
	nueek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(nueek, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *nueik;
	nueik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(nueik, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *isigPk;
	isigPk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(isigPk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *invnk;
	invnk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(invnk, 42, nx * nyk * sizeof(fftw_complex)); // test

	calcCollFreqk(nek, Tik, Tek, kb, eps0, mi, me, ri, rn, nn, Oci, Oce, e, nuink, nuiek, nuiik, nuenk, nueek, nueik, isigPk, invnk);

	// Calculate initial potential here
	fftw_complex *potSourcek;
	potSourcek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// initialize potSourceK, use memset for bits and for loop for values
	memset(potSourcek, 42, nx * nyk * sizeof(fftw_complex)); // test

	calcPotSourcek(dndxk, dndyk, Pik, Pek, nuink, nuiek, nuenk, nueik, isigPk, Oci, Oce, u, B, ksqu, potSourcek);

	int phi_iter = potentialk(invnk, dndxk, dndyk, phik, potSourcek, kx, ky, ninvksqu, err_max, phi_iter_max);

	// Initialize a time vector of length iter_max+1.
	std::vector<double> time(iter_max + 1, 0.0); // This will create a vector of size iter_max + 1 all initialized to 0.0. You could use memset as well

	c2rfft(nek, ne);
	c2rfft(Tek, Te);
	c2rfft(Tik, Ti);
	c2rfft(phik, phi);
	// Initialize velocities

	double *vexbx;
	vexbx = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vexbx, 42, nx * nyk * sizeof(double)); // added this here extra to testbed

	fftw_complex *vexbkx; // change name from vexbxk to vexbkx (defined differently in spectral funcs)
	vexbkx = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vexbkx, 42, nx * nyk * sizeof(fftw_complex));

	double *vexby;
	vexby = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vexby, 42, nx * nyk * sizeof(double)); // added this here extra to testbed since calc dt not tested

	fftw_complex *vexbky; // change name from vexbyk to vexbky (defined differently in spectral funcs)
	vexbky = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vexbky, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	double *vdmex;
	vdmex = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmex, 42, nx * nyk * sizeof(double));

	fftw_complex *vdmexk;
	vdmexk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmexk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	double *vdmey;
	vdmey = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmey, 42, nx * nyk * sizeof(double)); // added this here extra to testbed since calc dt not tested

	fftw_complex *vdmeyk;
	vdmeyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmeyk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	double *vdmix;
	vdmix = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmix, 42, nx * ny * sizeof(double)); // added this here extra to testbed since calc dt not tested

	fftw_complex *vdmixk;
	vdmixk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmixk, 42, nx * nyk * sizeof(fftw_complex));

	double *vdmiy;
	vdmiy = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmiy, 42, nx * ny * sizeof(double)); // added this here extra to testbed since calc dt not tested

	fftw_complex *vdmiyk;
	vdmiyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmiyk, 42, nx * nyk * sizeof(fftw_complex));

	calcV_ExBk(dphidxk, dphidyk, B, B2, vexbkx, vexbky);
	c2rfft(vexbkx, vexbx);
	c2rfft(vexbky, vexby);

	calc_diamag(dpedxk, dpedyk, B, B2, -1 * e, nek, vdmexk, vdmeyk);
	calc_diamag(dpidxk, dpidyk, B, B2, e, nek, vdmixk, vdmiyk);

	c2rfft(vdmexk, vdmex);
	c2rfft(vdmeyk, vdmey);
	c2rfft(vdmixk, vdmix);
	c2rfft(vdmiyk, vdmiy);

	fftw_complex *nek_old;
	nek_old = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *Tik_old;
	Tik_old = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *Tek_old;
	Tek_old = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *veoxk;
	veoxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(veoxk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	fftw_complex *veoyk;
	veoyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(veoyk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	fftw_complex *vioxk;
	vioxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vioxk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	fftw_complex *vioyk;
	vioyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vioyk, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *residualnk;
	residualnk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(residualnk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	fftw_complex *residualtik;
	residualtik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(residualtik, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *residualtek;
	residualtek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(residualtek, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	fftw_complex *sourcenk;
	sourcenk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(sourcenk, 42, nx * nyk * sizeof(fftw_complex));

	for (int iter = 0; iter < iter_max; iter++)
	{
		// 	for (int iter = 0; iter < iter_max; iter++){
		// 	for (int iter = 0; iter < 3; iter++){

		// Calculate pressures
		convolve2D(nek, Tek, Pek);
		rscalArr3DMult(Pek, kb, Pek);

		convolve2D(nek, Tik, Pik);
		rscalArr3DMult(Pik, kb, Pik);

		// c2rfft(Pek, Pe); //take inv
		// print2DPhysArray(Pe);

		// calc collision freq
		calcCollFreqk(nek, Tik, Tek, kb, eps0, mi, me, ri, rn, nn, Oci, Oce, e, nuink, nuiek, nuiik, nuenk, nueek, nueik, isigPk, invnk);

		// Calculate potential

		derivk(nek, kx, dndxk);
		derivk(nek, ky, dndyk);
		calcPotSourcek(dndxk, dndyk, Pik, Pek, nuink, nuiek, nuenk, nueik, isigPk, Oci, Oce, u, B, ksqu, potSourcek);

		phi_iter = potentialk(invnk, dndxk, dndyk, phik, potSourcek, kx, ky, ninvksqu, err_max, phi_iter_max);
		// Check for convergence with potential. We can go over this part together later.
		// if (phi_iter > phi_iter_max){
		//	printf("Blew up");
		// break;
		//}
		//	// Calculate all  velocities
		derivk(phik, kx, dphidxk);
		derivk(phik, ky, dphidyk);

		calcV_ExBk(dphidxk, dphidyk, B, B2, vexbkx, vexbky);
		c2rfft(vexbkx, vexbx);

		c2rfft(vexbky, vexby);
		// add more stuff

		derivk(Pek, kx, dpedxk);
		derivk(Pek, ky, dpedyk);
		derivk(Pik, kx, dpidxk);
		derivk(Pik, ky, dpidyk);

		calc_diamag(dpedxk, dpedyk, B, B2, -1 * e, nek, vdmexk, vdmeyk);
		calc_diamag(dpidxk, dpidyk, B, B2, e, nek, vdmixk, vdmiyk);

		c2rfft(vdmexk, vdmex);
		c2rfft(vdmeyk, vdmey);
		c2rfft(vdmixk, vdmix);
		c2rfft(vdmiyk, vdmiy);

		// Calculate time step
		double dt = calc_dt(u, vexbx, vexby, vdmix, vdmiy, vdmex, vdmey, CFL, kmax, dt_max);

		time[iter + 1] = time[iter] + dt;
		// Set ne_old = ne, Ti_old = Ti, Te_old = Te. This is because we need to save these for the RK method
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < nyk; j++)
			{
				for (int k = 0; k < ncomp; k++)
				{
					nek_old[j + nyk * i][k] = nek[j + nyk * i][k];
					Tek_old[j + nyk * i][k] = Tek[j + nyk * i][k];
					Tik_old[j + nyk * i][k] = Tik[j + nyk * i][k];
				}
			}
		}
		// Begin RK method

		for (int stage = 0; stage < 4; stage++)
		{
			// Calculate diamagnetic drifts
			convolve2D(nek, Tek, Pek);
			rscalArr3DMult(Pek, kb, Pek);

			convolve2D(nek, Tik, Pik);
			rscalArr3DMult(Pik, kb, Pik);

			derivk(Pek, kx, dpedxk);
			derivk(Pek, ky, dpedyk);
			derivk(Pik, kx, dpidxk);
			derivk(Pik, ky, dpidyk);

			calc_diamag(dpedxk, dpedyk, B, B2, -1 * e, nek, vdmexk, vdmeyk);
			calc_diamag(dpidxk, dpidyk, B, B2, e, nek, vdmixk, vdmiyk);

			// Get total velocity

			Arr3DArr3DAdd(vdmexk, vexbkx, veoxk);
			Arr3DArr3DAdd(vdmeyk, vexbky, veoyk);
			Arr3DArr3DAdd(vdmixk, vexbkx, vioxk);
			Arr3DArr3DAdd(vdmiyk, vexbkx, vioyk);

			// Get all residuals
			// c2rfft(vdmexk, vdmex);
			// print2DPhysArray(vdmex);

			calc_residualn(vexbkx, vexbky, nek, residualnk, kx, ky);

			calc_residualt(vioxk, vioyk, Tik, residualtik, kx, ky);

			calc_residualt(veoxk, veoyk, Tek, residualtek, kx, ky);

			// Get all source terms (only density for now is fine)

			calc_sourcen(ksqu, nek, Dart, sourcenk);

			// Update variables using RK method

			RK4(nek_old, dt, residualnk, sourcenk, stage, nek);

			RK4(Tik_old, dt, residualtik, sourcetk, stage, Tik);
		}
	}

	free(ne);
	free(nek);
	free(XX);
	free(YY);
	free(Pi);
	free(Pik);
	free(Pe);
	free(Pek);

	free(sourcetk);
	free(kx);
	free(ky);
	free(ksqu);
	free(ninvksqu);
	free(phi);
	free(phik);
	free(dndxk);
	free(dndyk);
	free(dphidxk);
	free(dphidyk);
	free(dpedxk);
	free(dpedyk);
	free(dpidxk);
	free(dpidyk);
	free(nuink);
	free(nuiek);
	free(nuiik);
	free(nuenk);
	free(nueek);
	free(nueik);
	free(isigPk);
	free(invnk);
	free(potSourcek);
	free(vexbx);
	free(vexbkx);
	free(vexby);
	free(vexbky);
	free(vdmex);
	free(vdmexk);
	free(vdmey);
	free(vdmeyk);
	free(vdmix);
	free(vdmixk);
	free(vdmiy);
	free(vdmiyk);
	free(nek_old);
	free(Tik_old);
	free(Tek_old);
	free(veoxk);
	free(veoyk);
	free(vioxk);
	free(vioyk);
	free(residualnk);
	free(residualtik);
	free(residualtek);
	free(sourcenk);

	// free(residualnk);
	// free(residualtik);
	// free(residualtek);
	// free(sourcenk);
	// free(sourcetk);
	// free(nek_old);
	// free(Tik_old);
	// free(Tek_old);

	// free more stuff
	printf("\n main() ends...\n");

	return 0;
}

void r2cfft(double rArr[], double cArr[][ncomp])
{
	// printf("\n Inside r2cfft() \n");
	fftw_plan r2c;
	// void fftw_plan_with_nthreads(int nbthreads); // before calling any plan OR: fftw_plan_with_nthreads(omp_get_max_threads())
	// #pragma omp critical (FFTW)
	// fftw_plan r2c;  // test for race condition?
	r2c = fftw_plan_dft_r2c_2d(nx, ny, &rArr[0], &cArr[0], FFTW_ESTIMATE);

	fftw_execute(r2c);

	//#pragma omp critical (FFTW)
	fftw_destroy_plan(r2c);

	fftw_cleanup();
	// Thread clean
	//  fftw_cleanup_threads();
	// you should create/destroy plans only from a single thread, but can safely execute multiple plans in parallel.
}

// This function multiplies a 2D physical array by a scalar. Division handled by just modifying scal outside of the function.
void scalArr2DMult(double arr[], double scal, double arrOut[])
{
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			arrOut[j + ny * i] = arr[j + ny * i] * scal;
		}
	}
}

void c2rfft(double cArr[][ncomp], double rArr[])
{
	// Make a dummy variable so we don't overwrite data
	fftw_complex *dummy; // TEST
	dummy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// Set this dummy variable equal to the complex array
	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				// fftw_complex *dummy; // for race condition test
				// dummy = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));
				dummy[j + nyk * i][k] = cArr[j + nyk * i][k];
			}
		}
	}
	// Make a FFT plan
	fftw_plan c2r;
// Define the FFT plan
#pragma omp critical(FFTW)
	// fftw_complex *dummy; // for race condition test
	// dummy = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));
	c2r = fftw_plan_dft_c2r_2d(nx, ny, &dummy[0], &rArr[0], FFTW_ESTIMATE); // dummy[0] is real part of dummy
	// Run FFT
	fftw_execute(c2r);

	fftw_destroy_plan(c2r);

	fftw_cleanup();

	scalArr2DMult(rArr, 1.0 / (nx * ny), rArr); // renormalize

	fftw_free(dummy);
}

void print2DPhysArray(double arr[])
{
	//#pragma omp parallel
	// instead of using two
	// #pragma omp parallel for collapse(2)
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		//#pragma omp parallel for
		for (int j = 0; j < ny; j++)
		{
			printf("%+4.2le  ", arr[j + ny * i]);
		}
		printf("\n");
	}
}

double makeSpatialMesh2D(double dx, double dy, double xarr[], double yarr[])
{
	// Make x coordinates. We want them to change as we change the zeroth index and be constant as we change the first index.
	//#pragma omp parallel

	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			xarr[j + ny * i] = i * dx;
			yarr[j + ny * i] = j * dy;
		}
	}

	if (dx < dy)
	{
		return 1 / (2 * dx);
	}
	else
	{
		return 1 / (2 * dy);
	}
}
void Arr3DArr2DMult(double arr3D[][ncomp], double arr2D[], double arrOut[][ncomp])
{
	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				arrOut[j + nyk * i][k] = arr3D[j + nyk * i][k] * arr2D[j + nyk * i];
			}
		}
	}
}
// This function multiplies a 3D Fourier array by i.
void iArr3DMult(double arr[][ncomp], double arrOut[][ncomp])
{
	// double dummy;   // This is used so that data are not accidentally overwritten
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			double dummy; // to fix race condition. The variable is written and read in parallel without any synchronization. Declaring the variable inside the innermost loop fixes it.
			dummy = arr[j + nyk * i][0];
			arrOut[j + nyk * i][0] = -arr[j + nyk * i][1];
			arrOut[j + nyk * i][1] = dummy;
		}
	}
}

void derivk(double vark[][ncomp], double k[], double derivative[][ncomp])
{
	// Multiply by the corresponding k
	Arr3DArr2DMult(vark, k, derivative);
	// Multiply by i
	iArr3DMult(derivative, derivative);
}

void makeFourierMesh2D(double Lx, double Ly, double KX[], double KY[], double ksqu[], double ninvksqu[], int FourierMeshType)
{
	// void makeFourierMesh2D(double Lx, double Ly, double KX[], double KY[], double ksqu[], double ninvksqu[]){
	// Make a variable to account for way wavenumbers are set in FFTW3.
	// int k_counter = 0;
	//	for(int i = 0; i < nx ; i++){
	//		for(int j = 0; j < nyk; j++){
	//			if (i < nx/2){ // i<nx/2 --> 0 to 127
	//			KX[j + nyk*i] =2.*M_PI*i/Lx; //K = 2pi/L* [0:nx/2-1	, -n/2:-1]' : creates a coloumn vector with nx/2 elements starting from 0 ( from 0 to 127 then -128 to -1) 256/2 is 128
	//			}
	//			if( i >= nx/2){ // i >= nx/2 --> from -128 to -1
	//			KX[j + nyk*i] =  2.* M_PI * (-i + 2.*k_counter) / Lx;
	//			}
	//		}
	//			if( i >= nx/2){
	//				k_counter++;
	//				}
	//	}
	// openmp: to avoid the "collapsed loops nested error" try simplifying this to two separated nested for loops and further simplify the is statements:

	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx / 2; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			KX[j + nyk * i] = 2. * M_PI * i / Lx;
		}
	}

	// for(int i = nx/2; int k_counter = 0; i < nx , i++, k_counter++){ gives error: expected primary-expression before ‘int’ ...
	//#pragma omp parallel for collapse(2)
	for (int i = nx / 2; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			int k_counter = i - nx / 2;
			KX[j + nyk * i] = 2. * M_PI * (-i + 2. * k_counter) / Lx; // You can then simplify `(-i + 2.*k_counter)` to `(-i + 2 * (  i - nx/2))` and then `(-i + 2*i - nx)`, which is `i - nx`.
		}
	}
	// print2DPhysArray(KX);
	// int k_counter = 0;
	//  Make kx. kx corresponds to modes [0:n/2-1 , -n/2:-1]. This is why there is an additional step, just due to FFTW3's structuring
	//#pragma omp parallel for collapse(2)
	// for(int i = 0; i < nx/2 ; i++){
	//	for(int j = 0; j < nyk; j++){
	// if (i < nx/2){ // i<nx/2 --> 0 to 127
	//			KX[j + nyk*i] =2.*M_PI*i/Lx; //K = 2pi/L* [0:nx/2-1	, -n/2:-1]' : creates a coloumn vector with nx/2 elements starting from 0 ( from 0 to 127 then -128 to -1) 256/2 is 128
	//}
	//		if( i >= nx/2){ // i >= nx/2 --> from -128 to -1
	//			KX[j + nyk*i] =  2.* M_PI * (-i + 2.*k_counter) / Lx;
	//		}
	//	}
	//	if( i >= nx/2){
	//		k_counter++;
	//	}
	//}

	// Make ky. Because we are using special FFTs for purely real valued functions, the last dimension (y in this case) only need n/2 + 1 points due to symmetry about the imaginary axis.
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			KY[j + nyk * i] = 2 * M_PI * j / Ly;
		}
	}
	// print2DPhysArray(KY);
	//  This is for the exact case = 0
	//  Make ksqu. k^2 = kx^2 + ky^2.
	//  Also make a variable for -1/k^2. This is used in potentialk

	double dx = Lx / nx; // test for race condition
	double dy = Ly / ny; // test
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			if (FourierMeshType == 0)
			{
				ksqu[j + nyk * i] = KX[j + nyk * i] * KX[j + nyk * i] + KY[j + nyk * i] * KY[j + nyk * i];
				ninvksqu[j + nyk * i] = -1 / ksqu[j + nyk * i];
				// ninvksqu[j + nyk*i] = -1 / (KX[j + nyk*i] * KX[j + nyk*i] + KY[j + nyk*i] * KY[j + nyk*i]);
			}
			if (FourierMeshType == 1)
			{
				// KXvec[]
				ksqu[j + nyk * i] = ((sin(KX[j + nyk * i] * dx / 2) / (dx / 2)) * (sin(KX[j + nyk * i] * dx / 2) / (dx / 2)) + (sin(KY[j + nyk * i] * dy / 2) / (dy / 2)) * (sin(KY[j + nyk * i] * dy / 2) / (dy / 2))); // Use approximations for Kx, Ky, K^2
				// ksqu[j + nyk*i] = (pow(sin(KX[j + nyk*i] * dx/2)/(dx/2),2) + pow(sin(KY[j + nyk*i] * dy/2)/(dy/2),2)) ;
				ninvksqu[j + nyk * i] = -1 / ksqu[j + nyk * i];
				// ninvksqu[j + nyk*i] = -1 /(pow(sin(KX[j + nyk*i] * dx/2)/(dx/2),2) + pow(sin(KY[j + nyk*i] * dy/2)/(dy/2),2));
				// ninvksqu[j + nyk*i] = -1 /((sin(KX[j + nyk*i] * dx/2)/(dx/2))*(sin(KX[j + nyk*i] * dx/2)/(dx/2)) + (sin(KY[j + nyk*i] * dy/2)/(dy/2))*(sin(KY[j + nyk*i] * dy/2)/(dy/2)));

				KX[j + nyk * i] = sin(KX[j + nyk * i] * dx) / dx; // overwrite the exact
				KY[j + nyk * i] = sin(KY[j + nyk * i] * dy) / dy;
			}
			// For the "approximation" case. the discrt k is not periodic so causes issues
		}
	}

	// Account for the [0][0] point since there is a divide by 0 issue.
	ninvksqu[0] = -1.;
	// print2DPhysArray(ksqu);
}
// This function calculates the magnitude of a 1D array with 3 elements
// added this not sure since contains 3 elements
double mag1DArray(double arr[])
{
	return sqrt(arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2]);
}

// Calculate collision frequencies
void calcCollFreqk(double nek[][ncomp], double Tik[][ncomp], double Tek[][ncomp], double kb, double eps0, double mi, double me, double ri, double rn, double nn, double Oci, double Oce, double e, double nuink[][ncomp], double nuiek[][ncomp], double nuiik[][ncomp], double nuenk[][ncomp], double nueek[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double invnk[][ncomp])
{ // Take inverse fft of nek, Tik, and Tek.
	// The reason we have to convert all of this to real space is because we can't take square roots or reciprocals in Fourier space

	//*********TEST for inertial add this in the function argument: double hallIk[][ncomp], double hallEk[][ncomp]*************
	// void calcCollFreqk( double nek[][ncomp], double Tik[][ncomp], double Tek[][ncomp] , double kb, double eps0, double mi, double me, double ri, double rn, double nn, double Oci, double Oce, double e, double nuink[][ncomp], double nuiek[][ncomp], double nuiik[][ncomp], double nuenk[][ncomp], double nueek[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double invnk[][ncomp], double hallIk[][ncomp], double hallEk[][ncomp]){

	double *ne; // TEST INSIDE AND OUTSIDE LOOP
	ne = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *Ti;
	Ti = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *Te;
	Te = (double *)fftw_malloc(nx * ny * sizeof(double));

	c2rfft(nek, ne);
	c2rfft(Tik, Ti);
	c2rfft(Tek, Te);

	// Set scalar doubles for variables that are needed in the loops
	double Vthi, Vthe, lambdaD, Lambda;

	// Initialize 2D physical arrays for the collision frequencies. Will take fft of them all at the end

	double *nuin; // test add inside and outside? for r2c func??
	nuin = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuii;
	nuii = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuie;
	nuie = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuen; // test add inside and outside? for r2c func??
	nuen = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuei;
	nuei = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuee; // test add inside and outside? for r2c func??
	nuee = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *isigP;
	isigP = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *invn;
	invn = (double *)fftw_malloc(nx * ny * sizeof(double));

	// Begin big loop to calculating everything.
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			// double Vthi, Vthe, lambdaD, Lambda; // for race condition
			// double *nuin; // TEST for race ..
			// nuin = (double*) fftw_malloc(nx*ny*sizeof(double));
			// double *nuen; // TEST for race ..
			// nuen= (double*) fftw_malloc(nx*ny*sizeof(double));
			// double *nuee; // TEST for race ..
			// nuee = (double*) fftw_malloc(nx*ny*sizeof(double));

			// double *ne; // TEST INSIDE AND OUTSIDE LOOP
			// ne = (double*) fftw_malloc(nx*ny*sizeof(double));

			// Calculate thermal velocities
			Vthi = sqrt(2. * kb * Ti[j + ny * i] / mi);
			Vthe = sqrt(2. * kb * Te[j + ny * i] / me);

			// Calculate ion-neutral and electron-neutral collision frequencies
			nuin[j + ny * i] = nn * Vthi * M_PI * (ri + rn) * (ri + rn);
			nuen[j + ny * i] = nn * Vthe * M_PI * rn * rn;

			// Calculate "inverse Pedersen conductivity"
			isigP[j + ny * i] = 1. / (e * (nuin[j + ny * i] / Oci + nuen[j + ny * i] / Oce));

			// Calculate Debye length
			lambdaD = sqrt(eps0 * kb * Te[j + ny * i] / (ne[j + ny * i] * e * e));

			// Calculate plasma parameter
			Lambda = 12. * M_PI * ne[j + ny * i] * lambdaD * lambdaD * lambdaD;

			// Calculate electron-electron collision frequency
			nuee[j + ny * i] = ne[j + ny * i] * e * e * e * e * log(Lambda / 3.) / (2. * M_PI * eps0 * eps0 * me * me * Vthe * Vthe * Vthe); // This time the difference is very significant. x * x * x is two orders of magnitude faster than std::pow(x, n)
			// nuee[j + ny*i] = ne[j + ny*i] * pow(e,4.) * log(Lambda/3.) / ( 2. * M_PI * eps0 * eps0 * me * me * pow(Vthe,3.)); // This time the difference is very significant. x * x * x is two orders of magnitude faster than std::pow(x, n)

			// Calculate ion-ion collision frequency
			nuii[j + ny * i] = nuee[j + ny * i] * sqrt(me / mi);

			// Calculate ion-electron collision frequency
			nuie[j + ny * i] = nuee[j + ny * i] * 0.5 * me / mi;

			// Calculate electron-ion collision frequency
			nuei[j + ny * i] = nuee[j + ny * i];

			// Calculate the inverse of the density
			// inverse of ne in Fourier space (which is needed for several terms in the temperature equation )
			invn[j + ny * i] = 1. / ne[j + ny * i];
		}
	}

	// Take FFTs of everything now
	r2cfft(nuin, nuink);
	r2cfft(nuie, nuiek);
	r2cfft(nuii, nuiik);
	r2cfft(nuen, nuenk);
	r2cfft(nuei, nueik);
	r2cfft(nuee, nueek);
	r2cfft(invn, invnk);
	r2cfft(isigP, isigPk);

	fftw_free(ne);
	fftw_free(Ti);
	fftw_free(Te);
	fftw_free(nuin);
	fftw_free(nuie);
	fftw_free(nuii);
	fftw_free(nuen);
	fftw_free(nuee);
	fftw_free(nuei);
	fftw_free(isigP);
	fftw_free(invn);
}

// This function multiplies, elementwise, a 2D physical array by a 2D physical array
void Arr2DArr2DMult(double arrIn0[], double arrIn1[], double arrOut[])
{
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			arrOut[j + ny * i] = arrIn0[j + ny * i] * arrIn1[j + ny * i];
		}
	}
}

void convolve2D(double fk[][ncomp], double gk[][ncomp], double fgk[][ncomp])
{
	// Initialize a variable f, g, and fg.

	double *f;
	f = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *g;
	g = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *fg;
	fg = (double *)fftw_malloc(nx * ny * sizeof(double));

	// Take inverse ffts
	c2rfft(fk, f);
	c2rfft(gk, g);

	// Multiply in real space
	Arr2DArr2DMult(f, g, fg);

	// Take fft of fg
	r2cfft(fg, fgk);

	fftw_free(f);
	fftw_free(g);
	fftw_free(fg);
}
// This function multiplies a 3D Fourier array by a real valued scalar. Handle division by modifying rscal outside of function.
void rscalArr3DMult(double arr[][ncomp], double rscal, double arrOut[][ncomp])
{
	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				arrOut[j + nyk * i][k] = arr[j + nyk * i][k] * rscal;
			}
		}
	}
}

// Take second derivative in Fourier space.
void laplaciank(double vark[][ncomp], double ksqu[], double derivative[][ncomp])
{
	// Multiply by k^2
	Arr3DArr2DMult(vark, ksqu, derivative);
	// Multiply by -1.
	rscalArr3DMult(derivative, -1., derivative);
	// Consider just haveing ksqu be -ksqu. Not sure where else this shows up and it might save a few operations.
}

// Calculate the source term in the potential equation
void calcPotSourcek(double dndxk[][ncomp], double dndyk[][ncomp], double Pik[][ncomp], double Pek[][ncomp], double nuink[][ncomp], double nuiek[][ncomp], double nuenk[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double Oci, double Oce, double u[], double B[], double ksqu[], double potSourcek[][ncomp])
{
	//
	// Get Laplacians of pressures

	fftw_complex *d2Pik;
	d2Pik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *d2Pek;
	d2Pek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	laplaciank(Pik, ksqu, d2Pik);
	laplaciank(Pek, ksqu, d2Pek);

	double *dummy;
	dummy = (double *)fftw_malloc(nx * ny * sizeof(double));

	c2rfft(d2Pik, dummy);
	// print2DPhysArray(dummy);

	// Calculate terms inside of parentheses for ions and electrons

	fftw_complex *pikTerm;
	pikTerm = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *pekTerm;
	pekTerm = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				pikTerm[j + nyk * i][k] = (nuiek[j + nyk * i][k] - nuink[j + nyk * i][k]) / Oci + nueik[j + nyk * i][k] / Oce;
				pekTerm[j + nyk * i][k] = nuiek[j + nyk * i][k] / Oci + (nueik[j + nyk * i][k] + nuenk[j + nyk * i][k]) / Oce;
			}
		}
	}

	// First convolution for ions to multiply by Laplacian of pressure
	convolve2D(d2Pik, pikTerm, pikTerm);

	// Second convolution for ions to multiply by isigP. This gives final pikTerm
	convolve2D(isigPk, pikTerm, pikTerm);

	// First convolution for electrons to multiply by Laplacian of pressure
	convolve2D(d2Pek, pekTerm, pekTerm);

	// Second convolution for electrons to multiply by isigP. This gives final pekTerm
	convolve2D(isigPk, pekTerm, pekTerm);

	// Calculate uxB term. For now, assume dndz = 0. Replace 0s in equations if needed in the future.
	// Then also in this loop, combine all of the terms together for the final source term
	double uxB_x, uxB_y, uxB_z, uxBTerm;
	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				uxB_x = B[0] * (dndyk[j + nyk * i][k] * u[2] - 0); // B[0] cause we start indexing from 0 in C unlike MATLAB
				uxB_y = B[1] * (0 - dndxk[j + nyk * i][k] * u[2]);
				uxB_z = B[2] * (dndxk[j + nyk * i][k] * u[1] - dndyk[j + nyk * i][k] * u[0]);
				uxBTerm = uxB_x + uxB_y + uxB_z;
				potSourcek[j + nyk * i][k] = uxBTerm + pikTerm[j + nyk * i][k] + pekTerm[j + nyk * i][k];
			}
		}
	}

	fftw_free(dummy);
	fftw_free(d2Pik);
	fftw_free(d2Pek);
	fftw_free(pikTerm);
	fftw_free(pekTerm);
}

// This function calculates the absolute value of a complex number. This is equivalent to the modulus.
void absComp(double arr3D[][ncomp], double arrOut[])
{
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			arrOut[j + nyk * i] = sqrt(arr3D[j + nyk * i][0] * arr3D[j + nyk * i][0] + arr3D[j + nyk * i][1] * arr3D[j + nyk * i][1]);
		}
	}
}

// This functions calculates the maximum value of a 2D Fourier array
// nx nyk size of complex 512x257 and real is nxny
double max2Dk(double arr2D[])
{
	double maxVal = 0.;
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			if (arr2D[j + nyk * i] > maxVal)
			{ // if (n)--> n isn't initialized here at
				// double *arr2D= new double [j+nyk*i]; //test this isn't initialized

				maxVal = arr2D[j + nyk * i];
			}
		}
	}
	return maxVal;
}

double max2D(double arr2D[])
{
	double maxVal = 0.;
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			if (arr2D[j + ny * i] > maxVal)
			{
				maxVal = arr2D[j + ny * i];
			}
		}
	}
	return maxVal;
}

double max_absComp(double arr3D[][ncomp])
{
	// Take the absolute value
	double *absArr;
	absArr = (double *)fftw_malloc(nx * nyk * sizeof(double));
	memset(absArr, 42, nx * nyk * sizeof(double)); // test here, if you pass 2D array to func decays to pointer and sizeof doesn't give size of array

	absComp(arr3D, absArr);

	// Calculate the max value
	double maxVal = max2Dk(absArr); // by
	fftw_free(absArr);

	return maxVal;
}

int potentialk(double invnk[][ncomp], double dndxk[][ncomp], double dndyk[][ncomp], double phik[][ncomp], double potSourcek[][ncomp], double kx[], double ky[], double ninvksqu[], double err_max, int max_iter)
{

	// function outputs phik and iterations ---> it solves for the potential using the spectral method

	// invnk is F[1/n] where F is fourier
	// potsourcek = F[s]
	// ninvksqu= 1/k^2

	// Initialize variables used in the function
	fftw_complex *dphidxk;
	dphidxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dphidyk;
	dphidyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *gradNgradPhi_x;
	gradNgradPhi_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *gradNgradPhi_y;
	gradNgradPhi_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *RHS;
	RHS = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	double phik_max, phik_max_old;
	double it_error;

	// Begin counter for the number of iterations it takes to calculate phi
	int count = 0;

	// Begin while loop
	do
	{
		// Calculate phi derivatives
		derivk(phik, kx, dphidxk);
		derivk(phik, ky, dphidyk);
		// Do convolutions for grad n dot grad phi term
		// gradNgradphi = [ikx phik] * [ikx nk] - [iky phik] * [iky nk], where * is the convolution
		convolve2D(dndxk, dphidxk, gradNgradPhi_x);
		convolve2D(dndyk, dphidyk, gradNgradPhi_y);

		// Subtract gradNgradphi from the source term. Calculate RHS of the equation:
		//#pragma omp parallel for collapse(3)
		for (int i = 0; i < nx; i++)
		{
			for (int j = 0; j < nyk; j++)
			{
				for (int k = 0; k < ncomp; k++)
				{
					RHS[j + nyk * i][k] = potSourcek[j + nyk * i][k] - gradNgradPhi_x[j + nyk * i][k] - gradNgradPhi_y[j + nyk * i][k];
				}
			}
		}

		// Convolve RHS with invnk
		convolve2D(RHS, invnk, RHS);

		// Calculate maximum of absolute value of previous phi
		phik_max_old = max_absComp(phik);

		// Multiply by ninvksqu to get the updated phi: this will output potential in Fourier space, phik
		Arr3DArr2DMult(RHS, ninvksqu, phik);

		// Calculate maximum of absolute value of updated phi(new phi)
		phik_max = max_absComp(phik); // by

		// Increase iteration count by 1
		count = count + 1;

		// Calculate error
		it_error = fabs((phik_max - phik_max_old) / phik_max); // err_max is the error we want to converge to

		// If error is too high and we haven't reached the max iterations yet, repeat iterations
	} while (it_error > err_max && count <= max_iter); // and instead &&

	fftw_free(dphidxk);
	fftw_free(dphidyk);
	fftw_free(gradNgradPhi_x);
	fftw_free(gradNgradPhi_y);
	fftw_free(RHS);
	// Output the number of iterations taken
	return count;
}

void print2DArrf(char filename[], double arr[])
{

	FILE *fptr;

	fptr = fopen(filename, "w");

	// do not know how parallelize this
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			fprintf(fptr, "%+4.2le  ", arr[j + ny * i]);
		}
		fprintf(fptr, "\n"); // because of this
	}

	fclose(fptr);
}

// Calculate the ExB drift
void calcV_ExBk(double dphidxk[][ncomp], double dphidyk[][ncomp], double B[], double B2, double vexbkx[][ncomp], double vexbky[][ncomp])
{ // Your inputs should be dphidxk, dphidyk, B, and B2
	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				vexbkx[j + nyk * i][k] = -1 * (dphidyk[j + nyk * i][k] * B[2]) / B2;
				vexbky[j + nyk * i][k] = (dphidxk[j + nyk * i][k] * B[2]) / B2;
			}
		}
	}
}

void Arr2DArr2DDiv(double arrIn0[], double arrIn1[], double arrOut[])
{
	//#pragma omp parallel for collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			arrOut[j + i * ny] = arrIn0[j + i * ny] / arrIn1[j + i * ny];
		}
	}
}

void fourierDivision2D(double fk[][ncomp], double gk[][ncomp], double fgk[][ncomp])
{
	// Initialize a variable f, g, and fg.
	double *f;
	f = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *g;
	g = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *fg;
	fg = (double *)fftw_malloc(nx * ny * sizeof(double));

	// Take inverse ffts
	c2rfft(fk, f);
	c2rfft(gk, g);

	// Multiply in real space
	Arr2DArr2DDiv(f, g, fg);

	// Take fft of fg
	r2cfft(fg, fgk);
	fftw_free(f);  // test was freed up in another function
	fftw_free(g);  // test was freed up in another function
	fftw_free(fg); // test was freed up in another function
}

void calc_diamag(double dpdxk[][ncomp], double dpdyk[][ncomp], double B[], double B2, double qa, double nak[][ncomp], double diamagxk[][ncomp], double diamagyk[][ncomp])
{

	fftw_complex *predivx;
	predivx = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *predivy;
	predivy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				predivx[j + nyk * i][k] = (dpdyk[j + nyk * i][k] * B[2]) / (B2 * qa * -1);
				predivy[j + nyk * i][k] = (dpdxk[j + nyk * i][k] * B[2]) / (B2 * qa);
			}
		}
	}

	fourierDivision2D(predivx, nak, diamagxk);
	fourierDivision2D(predivy, nak, diamagyk);

	fftw_free(predivx); // test
	fftw_free(predivy); // test
}

double Arr1DMax(double arr[], int arrLength)
{
	double max = 0.0;
	//#pragma omp parallel for
	for (int i = 0; i < arrLength; i++)
	{
		if (arr[i] > max)
		{
			max = arr[i];
		}
	}
	return max;
}

double calc_dt(double U[], double vexbx[], double vexby[], double diamagxi[], double diamagyi[], double diamagxe[], double diamagye[], double cfl, double kmax, double maxdt)
{

	// double bar = absolute(U);
	double *absArr;
	absArr = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(absArr, 42, nx*ny* sizeof(double));

	// for now this isn't needed. Will add it for the inertial code where the abs of u matters
	//#pragma omp parallel for
	// for (int i=0; i < size; i++){
	//	if (U[i] < 0){
	//		absArr[i] = abs(U[i]);

	//	}
	//}
	// print2DPhysArray(absArr);

	double vMaxArr[7];

	vMaxArr[0] = max2D(vexbx);
	vMaxArr[1] = max2D(vexby);
	vMaxArr[2] = max2D(diamagxi);
	vMaxArr[3] = max2D(diamagyi);
	vMaxArr[4] = max2D(diamagxe);
	vMaxArr[5] = max2D(diamagye);
	vMaxArr[6] = Arr1DMax(U, 3); // absolute of u, neutral wind //vMaxArr[6] = Arr1DMax((U), 3);
	// vMaxArr[6] = Arr1DMax(absArr,3);
	double max = Arr1DMax(vMaxArr, 7);

	double dt = cfl / (max * kmax); // added
	// try using min
	// dt = std::min(dt, maxdt);
	if (dt < maxdt)
	{
		return dt;
	}
	else
	{
		return maxdt;
	}
	fftw_free(absArr);
}

void Arr3DArr3DAdd(double arr0[][ncomp], double arr1[][ncomp], double arrOut[][ncomp])
{
	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				arrOut[j + nyk * i][k] = arr0[j + nyk * i][k] + arr1[j + nyk * i][k];
			}
		}
	}
}

void calc_residualn(double vexbxk[][ncomp], double vexbyk[][ncomp], double nink[][ncomp], double residnoutk[][ncomp], double kx[], double ky[])
{

	fftw_complex *dninxk;
	dninxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dninyk;
	dninyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *mult1;
	mult1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *mult2;
	mult2 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	derivk(nink, kx, dninxk);
	derivk(nink, ky, dninyk);

	convolve2D(vexbxk, dninxk, mult1);
	convolve2D(vexbyk, dninyk, mult2);

	Arr3DArr3DAdd(mult1, mult2, residnoutk);

	fftw_free(dninxk); // test no mult1/2 here
	fftw_free(dninyk);
	fftw_free(mult1); // test was freed up in another function
	fftw_free(mult2); // test was freed up in another function
}
void calc_residualt(double voxk[][ncomp], double voyk[][ncomp], double tempink[][ncomp], double tempoutk[][ncomp], double kx[], double ky[])
{

	fftw_complex *dtempinxk;
	dtempinxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dtempinyk;
	dtempinyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dvoxk;
	dvoxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dvoyk;
	dvoyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *divvo;
	divvo = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *mult1;
	mult1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *mult2;
	mult2 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *mult3;
	mult3 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dninxk;
	dninxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	derivk(tempink, kx, dtempinxk);
	derivk(tempink, ky, dtempinyk);
	derivk(voxk, kx, dvoxk);
	derivk(voyk, ky, dvoyk);

	Arr3DArr3DAdd(dvoxk, dvoyk, divvo);

	convolve2D(dtempinxk, voxk, mult1);
	convolve2D(dtempinyk, voyk, mult2);
	convolve2D(tempink, divvo, mult3);
	rscalArr3DMult(mult3, 2 / 3, mult3);

	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				tempoutk[j + nyk * i][k] = mult1[j + nyk * i][k] + mult2[j + nyk * i][k] + mult3[j + nyk * i][k];
			}
		}
	}
	fftw_free(dtempinxk); // test
	fftw_free(dtempinyk);
	fftw_free(dvoxk);
	fftw_free(dvoyk);
	fftw_free(divvo);
	fftw_free(mult1);
	fftw_free(mult2);
	fftw_free(mult3);
	fftw_free(dninxk);
}

void calc_sourcen(double ksqu[], double nk[][ncomp], double d, double sourcenk[][ncomp])
{

	fftw_complex *lapnk;
	lapnk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	laplaciank(nk, ksqu, lapnk);

	rscalArr3DMult(lapnk, d, sourcenk);

	fftw_free(lapnk); // test
}

void RK4(double f[][ncomp], double dt, double residual[][ncomp], double source[][ncomp], int stage, double fout[][ncomp])
{

	double alpha[4] = {1. / 4, 1. / 3, 1. / 2, 1.};
	//#pragma omp parallel for collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				fout[j + nyk * i][k] = f[j + nyk * i][k] - (alpha[stage] * dt * (residual[j + nyk * i][k] - source[j + nyk * i][k]));
			}
		}
	}
}

void print2DB(char filename[], double arr[])
{ // binary

	// open file to write solution
	FILE *fp = fopen(filename, "wb");
	if (!fp)
		return;

	// write grid
	uint64_t ndim = 2;
	uint64_t cells[] = {nx, ny};
	double lower[] = {0, 0}, upper[] = {1, 1}; //  test

	uint64_t real_type = 2;
	fwrite(&real_type, sizeof(uint64_t), 1, fp);

	fwrite(&ndim, sizeof(uint64_t), 1, fp);
	fwrite(cells, 2 * sizeof(uint64_t), 1, fp);
	fwrite(lower, 2 * sizeof(double), 1, fp);
	fwrite(upper, 2 * sizeof(double), 1, fp);

	uint64_t esznc = sizeof(double), size = nx * ny;
	fwrite(&esznc, sizeof(uint64_t), 1, fp);
	fwrite(&size, sizeof(uint64_t), 1, fp);

	fwrite(&arr[0], esznc * size, 1, fp);
	fclose(fp);
}