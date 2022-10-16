#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <stdio.h>
#include <math.h>
#include <string>
#include <cstring>
#include "spectralFunctions.h"
#include "arithmeticFunctions.h"
#include "fftw3.h" //instead of this include the file fftw_threads.h
//#include "threads/threads.h"// test
//#include<fftw_threads.h> //??
#include <valarray> // interesting stuff
#include <iostream> //for cout..
#include <vector>
#include <omp.h> //test

// This cpp file has all of the functions used for running the program.

// For FFTW threads use: --enable-threads in the flags and --enable-openmp

// Create the 2D spatial mesh, i.e. gives you the physical grid in x and y with 2D arrays XX and YY respectively
#define size 3
using namespace std;
double makeSpatialMesh2D(double dx, double dy, double xarr[], double yarr[])
{
	// Make x coordinates. We want them to change as we change the zeroth index and be constant as we change the first index.

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

// Get the frequency space coordinates. Also get ksqu for taking Laplacians. And get -1/ksqu modified for divide by 0 issues.
void makeFourierMesh2D(double Lx, double Ly, double KX[], double KY[], double ksqu[], double ninvksqu[], int FourierMeshType)
{
	// void makeFourierMesh2D(double Lx, double Ly, double KX[], double KY[], double ksqu[], double ninvksqu[]){
	// Make a variable to account for way wavenumbers are set in FFTW3.
	int k_counter = 0;
// Make kx. kx corresponds to modes [0:n/2-1 , -n/2:-1]. This is why there is an additional step, just due to FFTW3's structuring
#pragma omp parallel for schedule(dynamic) reduction(+ \
													 : k_counter)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			if (i < nx / 2)
			{										  // i<nx/2 --> 0 to 127
				KX[j + nyk * i] = 2. * M_PI * i / Lx; // K = 2pi/L* [0:nx/2-1	, -n/2:-1]' : creates a coloumn vector with nx/2 elements starting from 0 ( from 0 to 127 then -128 to -1) 256/2 is 128
			}
			if (i >= nx / 2)
			{ // i >= nx/2 --> from -128 to -1
				KX[j + nyk * i] = 2. * M_PI * (-i + 2. * k_counter) / Lx;
			}
		}
		if (i >= nx / 2)
		{
			k_counter++;
		}
	}

// Make ky. Because we are using special FFTs for purely real valued functions, the last dimension (y in this case) only need n/2 + 1 points due to symmetry about the imaginary axis.
#pragma omp parallel for schedule(static) collapse(2)
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
	double dx = Lx / nx; // test
	double dy = Ly / ny; // test
#pragma omp parallel for schedule(static) collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			if (FourierMeshType == 0)
			{
				ksqu[j + nyk * i] = KX[j + nyk * i] * KX[j + nyk * i] + KY[j + nyk * i] * KY[j + nyk * i];
				ninvksqu[j + nyk * i] = -1 / (KX[j + nyk * i] * KX[j + nyk * i] + KY[j + nyk * i] * KY[j + nyk * i]);
			}
			if (FourierMeshType == 1)
			{
				// KXvec[]
				ksqu[j + nyk * i] = (pow(sin(KX[j + nyk * i] * dx / 2) / (dx / 2), 2) + pow(sin(KY[j + nyk * i] * dy / 2) / (dy / 2), 2)); // Use approximations for Kx, Ky, K^2
				// pow((sin(KX[j + nyk*i] * dx/2))/dx/2,2) + pow((sin(KY[j + nyk*i] * dy/2))/dy/2,2);
				ninvksqu[j + nyk * i] = -1 / (pow(sin(KX[j + nyk * i] * dx / 2) / (dx / 2), 2) + pow(sin(KY[j + nyk * i] * dy / 2) / (dy / 2), 2));
				// 1 /(pow((sin(KX[j + nyk*i] * dx/2))/dx/2,2) + pow((sin(KY[j + nyk*i] * dy/2))/dy/2,2))
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
// openMP: when performing transform in FFTW, use one of these transform routines instead of the ordinary FFTW functions: fftw_threads(nthreads, plan, howmany, in, istride, idist,...)
// Make a real to complex fft function for ease of use. Use this to take forward FFTs.
void r2cfft(double rArr[], double cArr[][ncomp])
{
	// before creating a plan you want to parrallelize, call this function: void fftw_plan_with_nthreads(int nthreads);
	// Make a FFT plan
	// test fftw plan using openmp

	fftw_plan r2c;

	// Define the FFT plan
	r2c = fftw_plan_dft_r2c_2d(nx, ny, &rArr[0], &cArr[0], FFTW_ESTIMATE);

	fftw_execute(r2c);
	fftw_destroy_plan(r2c);
	fftw_cleanup();
	// Run FFT
}
//#pragma omp critical (FFTW) {
//	fftw_plan r2c;

// Make a c2r fft function for ease of use and to not lose data. Use this to take inverse FFTs.
// This function is important because of how FFTW3 does complex to real FFTs.
// Something about it internally causes the Fourier space variable to be overwritten.
// This is not something that is fine for us because we will need to take many inverse FFTs but still need the old values.
// So, this function put the variable into a dummy variable. Then, it takes the inverse FFT of that dummy variable so no data are lost.
void c2rfft(double cArr[][ncomp], double rArr[])
{
	// cout << "c2r start, thread: [" << omp_get_thread_num() << "]" << endl;
	double c2r_start_time = omp_get_wtime();

	// Make a dummy variable so we don't overwrite data
	fftw_complex *dummy;
	dummy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// Set this dummy variable equal to the complex array
	//#pragma omp parallel for schedule(static) collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				dummy[j + nyk * i][k] = cArr[j + nyk * i][k];
			}
		}
	}

// Define the FFT plan
// fftwf_plan_with_nthreads()
	fftw_plan c2r;
#pragma omp critical(FFTW)
	c2r = fftw_plan_dft_c2r_2d(nx, ny, &dummy[0], &rArr[0], FFTW_ESTIMATE); // dummy[0] is real part of dummy
	// Run FFT - this is thread safe
	fftw_execute(c2r);

#pragma omp critical(FFTW)
	{
		fftw_destroy_plan(c2r);
		fftw_cleanup();
	}

	// Need to account for normalization.
	// FFTW3's algorithm causes everything to be multiplied by nx*ny. We need to undo this and divide by nx*ny.
	// double rArr[16] = {0}; //added this not sure though

	scalArr2DMult(rArr, 1.0 / (nx * ny), rArr); // renormalize

	fftw_free(dummy);

	double c2r_end_time = omp_get_wtime();
	cout << "thread[" << omp_get_thread_num() << "] time taken for c2r in secs: " << c2r_end_time - c2r_start_time << endl;
}

// Take the derivative in Fourier space. The direction depends on what you use for k.
void derivk(double vark[][ncomp], double k[], double derivative[][ncomp])
{
	// Multiply by the corresponding k
	Arr3DArr2DMult(vark, k, derivative);
	// Multiply by i
	iArr3DMult(derivative, derivative);
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

// This is the convolution function. Think of this as multiplication in Fourier space.
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

// Calculate collision frequencies
void calcCollFreqk(double nek[][ncomp], double Tik[][ncomp], double Tek[][ncomp], double kb, double eps0, double mi, double me, double ri, double rn, double nn, double Oci, double Oce, double e, double nuink[][ncomp], double nuiek[][ncomp], double nuiik[][ncomp], double nuenk[][ncomp], double nueek[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double invnk[][ncomp])
{ // Take inverse fft of nek, Tik, and Tek.
	// The reason we have to convert all of this to real space is because we can't take square roots or reciprocals in Fourier space

	//*********TEST for inertial add this in the function argument: double hallIk[][ncomp], double hallEk[][ncomp]*************
	// void calcCollFreqk( double nek[][ncomp], double Tik[][ncomp], double Tek[][ncomp] , double kb, double eps0, double mi, double me, double ri, double rn, double nn, double Oci, double Oce, double e, double nuink[][ncomp], double nuiek[][ncomp], double nuiik[][ncomp], double nuenk[][ncomp], double nueek[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double invnk[][ncomp], double hallIk[][ncomp], double hallEk[][ncomp]){

	double *ne;
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

	double *nuin;
	nuin = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuii;
	nuii = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuie;
	nuie = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuen;
	nuen = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuei;
	nuei = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuee;
	nuee = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *isigP;
	isigP = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *invn;
	invn = (double *)fftw_malloc(nx * ny * sizeof(double));

// Begin big loop to calculating everything.
#pragma omp parallel for schedule(static) collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{

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
			nuee[j + ny * i] = ne[j + ny * i] * pow(e, 4.) * log(Lambda / 3.) / (2. * M_PI * eps0 * eps0 * me * me * pow(Vthe, 3.)); // This time the difference is very significant. x * x * x is two orders of magnitude faster than std::pow(x, n)

			// Calculate ion-ion collision frequency
			nuii[j + ny * i] = nuee[j + ny * i] * sqrt(me / mi);

			// Calculate ion-electron collision frequency
			nuie[j + ny * i] = nuee[j + ny * i] * 0.5 * me / mi;

			// Calculate electron-ion collision frequency
			nuei[j + ny * i] = nuee[j + ny * i];

			// Calculate the inverse of the density
			// inverse of ne in Fourier space (which is needed for several terms in the temperature equation )
			invn[j + ny * i] = 1. / ne[j + ny * i];

			//*************************	Hall parameters: TEST forb the inertial function ****************************
			// hallE[j + ny*i] =  nuen[j + ny*i]/Oce;
			// hallI[j + ny*i] = nuin[j + ny*i]/Oci;
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
	//***********test for the inertial function****************
	// r2cfft(hallE, hallEk);
	// r2cfft(hallI, hallIk);

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
	//************* TEST for inertial function*****************
	// fftw_free(hallE);
	// fftw_free(hallI);
}
// ************** INERTIAL TEST add this function ****************
// do I need new calcFreK_inertia function? yes

void calcCollFreqk_inertia(double nek[][ncomp], double Tik[][ncomp], double Tek[][ncomp], double kb, double eps0, double mi, double me, double ri, double rn, double nn, double Oci, double Oce, double e, double nuink[][ncomp], double nuiek[][ncomp], double nuiik[][ncomp], double nuenk[][ncomp], double nueek[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double invnk[][ncomp], double hallIk[][ncomp], double hallEk[][ncomp])
{ // Take inverse fft of nek, Tik, and Tek.
	// The reason we have to convert all of this to real space is because we can't take square roots or reciprocals in Fourier space

	//*********TEST for inertial add this in the function argument: double hallIk[][ncomp], double hallEk[][ncomp]*************
	// void calcCollFreqk( double nek[][ncomp], double Tik[][ncomp], double Tek[][ncomp] , double kb, double eps0, double mi, double me, double ri, double rn, double nn, double Oci, double Oce, double e, double nuink[][ncomp], double nuiek[][ncomp], double nuiik[][ncomp], double nuenk[][ncomp], double nueek[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double invnk[][ncomp], double hallIk[][ncomp], double hallEk[][ncomp]){

	double *ne;
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

	double *nuin;
	nuin = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuii;
	nuii = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuie;
	nuie = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuen;
	nuen = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuei;
	nuei = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *nuee;
	nuee = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *isigP;
	isigP = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *invn;
	invn = (double *)fftw_malloc(nx * ny * sizeof(double));
	//**********TEST for inertial function: add hallI and hallE here***************

	double *hallE;
	hallE = (double *)fftw_malloc(nx * ny * sizeof(double));

	double *hallI;
	hallI = (double *)fftw_malloc(nx * ny * sizeof(double));

// Begin big loop to calculating everything.
#pragma omp parallel for schedule(static) collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{

			// const int idx = j + ny*i;
			//  Calculate thermal velocities
			Vthi = sqrt(2. * kb * Ti[j + ny * i] / mi);
			; // Vthi = (2*kb * Ti / mi).^.5;
			// attempt1: Vthi = sqrt( 2. * kb / mi);
			// Vthi = sqrt( 2. * kb * Ti[j + ny*i] / mi);
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
			nuee[j + ny * i] = ne[j + ny * i] * pow(e, 4.) * log(Lambda / 3.) / (2. * M_PI * eps0 * eps0 * me * me * pow(Vthe, 3.));

			// Calculate ion-ion collision frequency
			nuii[j + ny * i] = nuee[j + ny * i] * sqrt(me / mi);

			// Calculate ion-electron collision frequency
			nuie[j + ny * i] = nuee[j + ny * i] * 0.5 * me / mi;

			// Calculate electron-ion collision frequency
			nuei[j + ny * i] = nuee[j + ny * i];

			// Calculate the inverse of the density
			// inverse of ne in Fourier space (which is needed for several terms in the temperature equation )
			invn[j + ny * i] = 1. / ne[j + ny * i];

			//*************************	Hall parameters: TEST forb the inertial function ****************************
			hallE[j + ny * i] = nuen[j + ny * i] / Oce;
			hallI[j + ny * i] = nuin[j + ny * i] / Oci;
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
	//***********test for the inertial function****************
	r2cfft(hallE, hallEk);
	r2cfft(hallI, hallIk);

	// test
	//  print debye length
	/*
	char Nuin[] = "Nuin.txt";
	print2DArrf(Nuin, nuin);

	char Nuie[] = "Nuie.txt";
	print2DArrf(Nuie, nuie);

	char Nuii[] = "Nuii.txt";
	print2DArrf(Nuii, nuii);

	char Nuen[] = "Nuen.txt";
	print2DArrf(Nuen, nuen);

	char Nuee[] = "Nuee.txt";
	print2DArrf(Nuee, nuee);

	char Nuei[] = "Nuei.txt";
	print2DArrf(Nuei, nuei);
	*/

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
	//************* TEST for inertial function*****************
	fftw_free(hallE);
	fftw_free(hallI);
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

#pragma omp parallel for schedule(static) collapse(3)
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
#pragma omp parallel for schedule(static) collapse(3)
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
				// printf("(%.15f, %.15f)\n", (*potSourcek)[0], (*potSourcek)[1]); //test to compare
			}
		}
	}

	fftw_free(dummy);
	fftw_free(d2Pik);
	fftw_free(d2Pek);
	fftw_free(pikTerm);
	fftw_free(pekTerm);
}
// TEST for inertial code
// from MATLAB: calcSourcePotential_inertia_phi1 and here we could call it calcPotSourcek_inertia
// maybe this should be under the potential function
void calcPotSourcek_inertia(double ne0k[][ncomp], double nek[][ncomp], double dndx0k[][ncomp], double dndy0k[][ncomp], double dndxk[][ncomp], double dndyk[][ncomp], double dphidx0k[][ncomp], double dphidy0k[][ncomp], double dphidx1k[][ncomp], double dphidy1k[][ncomp], double Pi1k[][ncomp], double Pe1k[][ncomp], double uxB[], double e, double Cm, double hallEk[][ncomp], double hallIk[][ncomp], double vexbkx0[][ncomp], double vexbky0[][ncomp], double vexbkx[][ncomp], double vexbky[][ncomp], double kx[], double ky[], double ksqu[], double potSourcek_inertia[][ncomp])
{

	// Neglect pressure terms for now
	// get Laplacians of ion pressure:
	fftw_complex *d2Pi1k;
	d2Pi1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	// get Laplacian of electron pressure:
	fftw_complex *d2Pe1k;
	d2Pe1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	// 2nd derivatives in Fourier space:
	laplaciank(Pi1k, ksqu, d2Pi1k); // not sure about d2pi1k
	laplaciank(Pe1k, ksqu, d2Pe1k);

	// calculate ion and electron pressure terms:

	fftw_complex *pikTerm;
	pikTerm = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *pekTerm;
	pekTerm = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

#pragma omp parallel for schedule(static) collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				pikTerm[j + nyk * i][k] = -1. * (Cm / e) * hallIk[j + nyk * i][k]; // not sure about Cm here
				pekTerm[j + nyk * i][k] = (Cm / e) * hallEk[j + nyk * i][k];
			}
		}
	}

	// First convolution for ions to multiply by Laplacian of pressure
	convolve2D(d2Pi1k, pikTerm, pikTerm); // this is = convolution2D of hallIk with -ksqu.*Pik

	// First convolution for electrons to multiply by Laplacian of pressure
	convolve2D(d2Pe1k, pekTerm, pekTerm);

	// Define more variables (defined already in main)
	fftw_complex *ne1k; // for density perturbations
	ne1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dndx1k; // this is for the density perturbations
	dndx1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dndy1k; // this is for the density perturbations
	dndy1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dphidxk;
	dphidxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dphidyk;
	dphidyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *vexbkx1; // change name from vexbxk to vexbkx (defined differently in spectral funcs)
	vexbkx1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *vexbky1; // change name from vexbxk to vexbkx (defined differently in spectral funcs)
	vexbky1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

// Get density perturbations: try a foor loop
#pragma omp parallel for schedule(static) collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				ne1k[j + nyk * i][k] = nek[j + nyk * i][k] - ne0k[j + nyk * i][k]; // not sure about [j + nyk*i][k]
				dndx1k[j + nyk * i][k] = dndxk[j + nyk * i][k] - dndx0k[j + nyk * i][k];
				dndy1k[j + nyk * i][k] = dndyk[j + nyk * i][k] - dndy0k[j + nyk * i][k];

				dphidxk[j + nyk * i][k] = dphidx0k[j + nyk * i][k] + dphidx1k[j + nyk * i][k];
				dphidyk[j + nyk * i][k] = dphidy0k[j + nyk * i][k] + dphidy1k[j + nyk * i][k];

				vexbkx1[j + nyk * i][k] = vexbkx[j + nyk * i][k] - vexbkx0[j + nyk * i][k];
				vexbky1[j + nyk * i][k] = vexbky[j + nyk * i][k] - vexbky0[j + nyk * i][k];
			}
		}
	}

	// divergence (n nabla phi) term:

	fftw_complex *div_n_nabphi_x;
	div_n_nabphi_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *div_n_nabphi_y;
	div_n_nabphi_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *div_dummy_x1;
	div_dummy_x1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *div_dummy_y1;
	div_dummy_y1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *div_dummy_x0;
	div_dummy_x0 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *div_dummy_y0;
	div_dummy_y0 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *div_dummyTerm;
	div_dummyTerm = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	//**********************************************************
	fftw_complex *div_dummy_dx1; // derivative of dummy wrt k
	div_dummy_dx1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *div_dummy_dy1; // derivative of dummy wrt k
	div_dummy_dy1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *div_dummy_dx0; // derivative of dummy wrt k
	div_dummy_dx0 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *div_dummy_dy0; // derivative of dummy wrt k
	div_dummy_dy0 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	//	fftw_complex *phi1k;
	//	phi1k = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));

	//	fftw_complex *phi0k;
	//	phi0k = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));
	// test this term this way since can't take convov of a sum
	fftw_complex *coeff;
	coeff = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	// Calculate phi derivatives
	//	derivk(phi1k, kx, dphidx1k); //not sure but phi1k since deriv is in main code
	//	derivk(phi1k, ky, dphidy1k);

	//	derivk(phi0k, kx, dphidx0k); //not sure but phi0k since deriv is in main code
	//	derivk(phi0k, ky, dphidy0k);
	// Do the convolution of nek and dphi1k , to find the sum of final div dummy term seperate them like this?
	convolve2D(nek, dphidx1k, div_dummy_x1);
	convolve2D(nek, dphidy1k, div_dummy_y1);

	convolve2D(ne1k, dphidx0k, div_dummy_x0);
	convolve2D(ne1k, dphidy0k, div_dummy_y0);

	// Take derivatives of dummy variable:
	derivk(div_dummy_x1, kx, div_dummy_dx1); // not sure
	derivk(div_dummy_y1, ky, div_dummy_dy1);

	derivk(div_dummy_x0, kx, div_dummy_dx0); // not sure
	derivk(div_dummy_y0, ky, div_dummy_dy0);

// then try for loop to sum two different terms?
#pragma omp parallel for schedule(static) collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				div_dummyTerm[j + nyk * i][k] = div_dummy_x1[j + nyk * i][k] + div_dummy_y1[j + nyk * i][k] + div_dummy_x0[j + nyk * i][k] + div_dummy_y0[j + nyk * i][k]; // not sure about [j + nyk*i][k]
				coeff[j + nyk * i][k] = -1. * (Cm) * (hallEk[j + nyk * i][k] + hallIk[j + nyk * i][k]);																	   // not sure how to do this

				div_n_nabphi_x[j + nyk * i][k] = div_dummy_dx1[j + nyk * i][k] + div_dummy_dx0[j + nyk * i][k]; // change x =x+..
				div_n_nabphi_y[j + nyk * i][k] = div_dummy_dy1[j + nyk * i][k] + div_dummy_dy0[j + nyk * i][k];
			}
		}
	}
	convolve2D(div_n_nabphi_x, coeff, div_n_nabphi_x);
	convolve2D(div_n_nabphi_y, coeff, div_n_nabphi_y);
	//
	// uXb Term: do the same thing with coeff here but different sign
	//  double uxBTerm; to find convov of this term then it must be complex
	//(think of uxBTerm as a place holder being updated at each iteration)

	fftw_complex *coeff1;
	coeff1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *uxBTerm;
	uxBTerm = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	// double B[3] = {0,0,5E-5};
	// double u[3] = {-500, 300, 0}; //if I define uxB here I get phi iter =2 at iteration 3 then back to 1??
#pragma omp parallel for schedule(static) collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				uxBTerm[j + nyk * i][k] = uxB[0] * dndx1k[j + nyk * i][k] + uxB[1] * dndy1k[j + nyk * i][k]; // no need uxBTerm =UxBTerm+...
				coeff1[j + nyk * i][k] = (Cm) * (hallEk[j + nyk * i][k] + hallIk[j + nyk * i][k]);			 // not sure how to do this
			}
		}
	}
	// coeff1 = (hallEk) + (hallIk);
	// Convolve uxBTerm with hallEk+hallIk
	convolve2D(uxBTerm, coeff1, uxBTerm);
	// *****************************************************
	// v nabla phi term:
	fftw_complex *v_nabnab_phiTerm; // derivative of dummy wrt k
	v_nabnab_phiTerm = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *nabnab_dummyTerm_n0v0phi1_x;
	nabnab_dummyTerm_n0v0phi1_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	fftw_complex *nabnab_dummyTerm_n0v0phi1_y;
	nabnab_dummyTerm_n0v0phi1_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	//************* First calculate n0v0phi1 term ***********
	fftw_complex *nabnab_dummyTerm_n0v0phi1; // use it for the sum of x and y
	nabnab_dummyTerm_n0v0phi1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *nabnab_dummyTerm_n1vphi_x;
	nabnab_dummyTerm_n1vphi_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	fftw_complex *nabnab_dummyTerm_n1vphi_y;
	nabnab_dummyTerm_n1vphi_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// cross terms
	fftw_complex *nabnab_dummyTerm_n1vphi_xy;
	nabnab_dummyTerm_n1vphi_xy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	fftw_complex *nabnab_dummyTerm_n1vphi_yx;
	nabnab_dummyTerm_n1vphi_yx = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	//************* Calculate other n1vphi term ***********
	fftw_complex *nabnab_dummyTerm_n1vphi;
	nabnab_dummyTerm_n1vphi = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *nabnab_dummyTerm_n0v1phi_x;
	nabnab_dummyTerm_n0v1phi_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	fftw_complex *nabnab_dummyTerm_n0v1phi_y;
	nabnab_dummyTerm_n0v1phi_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	//************* Calculate other n0v1phi term ***********
	fftw_complex *nabnab_dummyTerm_n0v1phi;
	nabnab_dummyTerm_n0v1phi = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	// Derivatives initialized:
	// laplacian of perturbed phi:
	fftw_complex *d2phidx1k; // initialize this in the main code as well?
	d2phidx1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *d2phidy1k;
	d2phidy1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *d2phidxk; // initialize this in the main code as well?
	d2phidxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *d2phidyk;
	d2phidyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// cross derivatives
	fftw_complex *d2phidxyk;
	d2phidxyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	fftw_complex *d2phidyxk;
	d2phidyxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *d2phidxy1k;
	d2phidxy1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	fftw_complex *d2phidyx1k;
	d2phidyx1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *nabnab_dummyTerm_n0v0phi1_xy;
	nabnab_dummyTerm_n0v0phi1_xy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *nabnab_dummyTerm_n0v0phi1_yx;
	nabnab_dummyTerm_n0v0phi1_yx = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *nabnab_dummyTerm_n0v1phi_xy;
	nabnab_dummyTerm_n0v1phi_xy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *nabnab_dummyTerm_n0v1phi_yx;
	nabnab_dummyTerm_n0v1phi_yx = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *nabnab_dummyTerm_n0v0phi1_yy; // FREE THIS
	nabnab_dummyTerm_n0v0phi1_yy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	fftw_complex *nabnab_dummyTerm_n1vphi_yy; // FREE THIS
	nabnab_dummyTerm_n1vphi_yy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	fftw_complex *nabnab_dummyTerm_n0v1phi_yy; // FREE THIS
	nabnab_dummyTerm_n0v1phi_yy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	// first term:

	// Derivatives:
	derivk(dphidx1k, kx, d2phidx1k);
	derivk(dphidy1k, ky, d2phidy1k);
	// cross deriv of dphidxk
	derivk(dphidx1k, ky, d2phidxy1k);
	derivk(dphidy1k, kx, d2phidyx1k);

	// convolutions:
	convolve2D(vexbkx0, d2phidx1k, nabnab_dummyTerm_n0v0phi1_x);
	convolve2D(vexbky0, d2phidy1k, nabnab_dummyTerm_n0v0phi1_y);

	// cross terms convo
	convolve2D(vexbky0, d2phidxy1k, nabnab_dummyTerm_n0v0phi1_xy);
	convolve2D(vexbkx0, d2phidyx1k, nabnab_dummyTerm_n0v0phi1_yx);

	// Calculate other n1vphi term:
	// Derivatives:
	derivk(dphidxk, kx, d2phidxk);
	derivk(dphidyk, ky, d2phidyk);
	// cross deriv of dphidxk
	derivk(dphidxk, ky, d2phidxyk);
	derivk(dphidyk, kx, d2phidyxk);

	// convolutions:
	convolve2D(vexbkx, d2phidxk, nabnab_dummyTerm_n1vphi_x);
	convolve2D(vexbky, d2phidyk, nabnab_dummyTerm_n1vphi_y);

	// cross terms convo
	convolve2D(vexbky, d2phidxyk, nabnab_dummyTerm_n1vphi_xy);
	convolve2D(vexbkx, d2phidyxk, nabnab_dummyTerm_n1vphi_yx);

	//  Calculate other n0v1phi term:
	// convolutions:
	convolve2D(vexbkx1, d2phidxk, nabnab_dummyTerm_n0v1phi_x);
	convolve2D(vexbky1, d2phidyk, nabnab_dummyTerm_n0v1phi_y);

	// cross terms convo
	convolve2D(vexbky1, d2phidxyk, nabnab_dummyTerm_n0v1phi_xy);
	convolve2D(vexbkx1, d2phidyxk, nabnab_dummyTerm_n0v1phi_yx);

#pragma omp parallel for schedule(static) collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				nabnab_dummyTerm_n0v0phi1[j + nyk * i][k] = (nabnab_dummyTerm_n0v0phi1_x[j + nyk * i][k]) + (nabnab_dummyTerm_n0v0phi1_xy[j + nyk * i][k]);
				nabnab_dummyTerm_n0v0phi1_yy[j + nyk * i][k] = (nabnab_dummyTerm_n0v0phi1_yx[j + nyk * i][k]) + (nabnab_dummyTerm_n0v0phi1_y[j + nyk * i][k]);
				nabnab_dummyTerm_n1vphi[j + nyk * i][k] = (nabnab_dummyTerm_n1vphi_x[j + nyk * i][k]) + (nabnab_dummyTerm_n1vphi_xy[j + nyk * i][k]);
				nabnab_dummyTerm_n1vphi_yy[j + nyk * i][k] = (nabnab_dummyTerm_n1vphi_yx[j + nyk * i][k]) + (nabnab_dummyTerm_n1vphi_y[j + nyk * i][k]);
				nabnab_dummyTerm_n0v1phi[j + nyk * i][k] = (nabnab_dummyTerm_n0v1phi_x[j + nyk * i][k]) + (nabnab_dummyTerm_n0v1phi_xy[j + nyk * i][k]);
				nabnab_dummyTerm_n0v1phi_yy[j + nyk * i][k] = (nabnab_dummyTerm_n0v1phi_yx[j + nyk * i][k]) + (nabnab_dummyTerm_n0v1phi_y[j + nyk * i][k]);
			}
		}
	}
	// Multiply by n0:
	// convolve2D(ne0k,nabnab_dummyTerm_n0v0phi1 ,nabnab_dummyTerm_n0v0phi1);
	convolve2D(nabnab_dummyTerm_n0v0phi1, ne0k, nabnab_dummyTerm_n0v0phi1);
	convolve2D(nabnab_dummyTerm_n0v0phi1_yy, ne0k, nabnab_dummyTerm_n0v0phi1_yy);

	// Multiply by n1:
	// convolve2D( ne1k, nabnab_dummyTerm_n1vphi,nabnab_dummyTerm_n1vphi);
	convolve2D(nabnab_dummyTerm_n1vphi, ne1k, nabnab_dummyTerm_n1vphi);
	convolve2D(nabnab_dummyTerm_n1vphi_yy, ne1k, nabnab_dummyTerm_n1vphi_yy);

	// Multiply by n0:
	convolve2D(ne0k, nabnab_dummyTerm_n0v1phi, nabnab_dummyTerm_n0v1phi);
	convolve2D(nabnab_dummyTerm_n0v1phi_yy, ne0k, nabnab_dummyTerm_n0v1phi_yy);

	fftw_complex *dnabnab_dummyTerm_n0v0phi1_x; // derivative of dummy wrt kx
	dnabnab_dummyTerm_n0v0phi1_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dnabnab_dummyTerm_n0v0phi1_y; // derivative of dummy wrt ky
	dnabnab_dummyTerm_n0v0phi1_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dnabnab_dummyTerm_n1vphi_x;
	dnabnab_dummyTerm_n1vphi_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dnabnab_dummyTerm_n1vphi_y;
	dnabnab_dummyTerm_n1vphi_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dnabnab_dummyTerm_n0v1phi_x;
	dnabnab_dummyTerm_n0v1phi_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dnabnab_dummyTerm_n0v1phi_y;
	dnabnab_dummyTerm_n0v1phi_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// cross terms
	fftw_complex *dnabnab_dummyTerm_n0v0phi1_xy;
	dnabnab_dummyTerm_n0v0phi1_xy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	fftw_complex *dnabnab_dummyTerm_n0v0phi1_yx;
	dnabnab_dummyTerm_n0v0phi1_yx = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dnabnab_dummyTerm_n1vphi_xy;
	dnabnab_dummyTerm_n1vphi_xy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dnabnab_dummyTerm_n1vphi_yx;
	dnabnab_dummyTerm_n1vphi_yx = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dnabnab_dummyTerm_n0v1phi_xy;
	dnabnab_dummyTerm_n0v1phi_xy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dnabnab_dummyTerm_n0v1phi_yx;
	dnabnab_dummyTerm_n0v1phi_yx = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	//*****************************
	// try nabnab_dummyTerm_n0v0phi1
	derivk(nabnab_dummyTerm_n0v0phi1, kx, dnabnab_dummyTerm_n0v0phi1_x);
	// derivk(nabnab_dummyTerm_n0v0phi1_x, kx, dnabnab_dummyTerm_n0v0phi1_x);
	derivk(nabnab_dummyTerm_n0v0phi1_yy, ky, dnabnab_dummyTerm_n0v0phi1_y);
	// derivk(nabnab_dummyTerm_n0v0phi1_y, ky, dnabnab_dummyTerm_n0v0phi1_y);

	// cross terms
	// derivk(nabnab_dummyTerm_n0v0phi1_x, ky, dnabnab_dummyTerm_n0v0phi1_xy);
	// derivk(nabnab_dummyTerm_n0v0phi1_y, kx, dnabnab_dummyTerm_n0v0phi1_yx);

	derivk(nabnab_dummyTerm_n1vphi, kx, dnabnab_dummyTerm_n1vphi_x);
	derivk(nabnab_dummyTerm_n1vphi_yy, ky, dnabnab_dummyTerm_n1vphi_y);
	// cross terms
	// derivk(nabnab_dummyTerm_n1vphi_x, ky, dnabnab_dummyTerm_n1vphi_xy);
	// derivk(nabnab_dummyTerm_n1vphi_y, kx, dnabnab_dummyTerm_n1vphi_yx);

	// derivk(nabnab_dummyTerm_n0v1phi_x, kx, dnabnab_dummyTerm_n0v1phi_x);
	derivk(nabnab_dummyTerm_n0v1phi, kx, dnabnab_dummyTerm_n0v1phi_x); // it was kx
	derivk(nabnab_dummyTerm_n0v1phi_yy, ky, dnabnab_dummyTerm_n0v1phi_y);

// it is the deriv of the terms AFTER convo no x,y,z components

// Sum the three terms together and take divergence :
#pragma omp parallel for schedule(static) collapse(2)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				v_nabnab_phiTerm[j + nyk * i][k] = -(dnabnab_dummyTerm_n0v0phi1_x[j + nyk * i][k] + dnabnab_dummyTerm_n0v0phi1_y[j + nyk * i][k] + dnabnab_dummyTerm_n1vphi_x[j + nyk * i][k] + dnabnab_dummyTerm_n1vphi_y[j + nyk * i][k] + dnabnab_dummyTerm_n0v1phi_x[j + nyk * i][k] + dnabnab_dummyTerm_n0v1phi_y[j + nyk * i][k]);
				//- (dnabnab_dummyTerm_n0v0phi1_x[j + nyk*i][k] + dnabnab_dummyTerm_n0v0phi1_y[j + nyk*i][k] + dnabnab_dummyTerm_n1vphi_x[j + nyk*i][k] + dnabnab_dummyTerm_n1vphi_y[j + nyk*i][k] + dnabnab_dummyTerm_n0v1phi_x[j + nyk*i][k] + dnabnab_dummyTerm_n0v1phi_y[j + nyk*i][k]);
				// potSourcek_inertia[j + nyk*i][k] =  hallEk[j + nyk*i][k]+ hallIk[j + nyk*i][k];
				potSourcek_inertia[j + nyk * i][k] = v_nabnab_phiTerm[j + nyk * i][k] + div_n_nabphi_x[j + nyk * i][k] + div_n_nabphi_y[j + nyk * i][k] + pikTerm[j + nyk * i][k] + pekTerm[j + nyk * i][k] + uxBTerm[j + nyk * i][k];

				// potSourcek_inertia[j + nyk*i][k] = v_nabnab_phiTerm[j + nyk*i][k] + div_n_nabphi_x[j + nyk*i][k]+div_n_nabphi_y[j + nyk*i][k] +  pikTerm[j + nyk*i][k] + pekTerm[j + nyk*i][k] + uxBTerm[j + nyk*i][k];

				//
				// potSourcek_inertia[j + nyk*i][k] = div_n_nabphi_x[j + nyk*i][k]+ div_n_nabphi_y[j + nyk*i][k] + uxBTerm[j + nyk*i][k]+ v_nabnab_phiTerm[j + nyk*i][k]+ pikTerm[j + nyk*i][k] + pekTerm[j + nyk*i][k];
				// printf("(%.15f, %.15f)\n", (*potSourcek_inertia)[0], (*potSourcek_inertia)[1]); //the fftw complex type holds a complex # as a 2 elment array of doubles, [0] accesses 1st element (real part) and [1] accesses the imaginary part

				// potSourcek_inertia is a pointer to such an array which is why use *
				// try print3DFourierArray or take inv and print real solts
				// take inv FFT of potsourcek inertia
			}
		}
	}

	double *potSource_inertia; //= (double**)malloc(sizeof (double*)) * (i-1)); //init here for test
	potSource_inertia = (double *)fftw_malloc(nx * ny * sizeof(double));
	// p
	/*
	c2rfft(potSourcek_inertia, potSource_inertia); //take inv
	//print2DPhysArray(potSource_inertia); //try *
	char potSourceI[] = "potSourceI.txt";
	print2DArrf(potSourceI, potSource_inertia);
	*/
	// compare my sourk to potsourcek_exact in MATLAB file potentialSolver_Test rigerous
	// if not same:1. make sure BG variables match the mathematica outputs
	// 2. if not working: go into source term and ckeck each term separately, e.g. pikterm; or so on
	// print out sourcek here and compare
	// free everything
	fftw_free(pikTerm);
	fftw_free(pekTerm);
	//
	fftw_free(dnabnab_dummyTerm_n0v1phi_y);
	fftw_free(dnabnab_dummyTerm_n0v1phi_x);
	fftw_free(dnabnab_dummyTerm_n1vphi_y);
	fftw_free(dnabnab_dummyTerm_n1vphi_x);
	fftw_free(dnabnab_dummyTerm_n0v0phi1_y);
	fftw_free(dnabnab_dummyTerm_n0v0phi1_x);
	fftw_free(d2phidy1k);
	fftw_free(d2phidx1k);
	fftw_free(nabnab_dummyTerm_n0v1phi);
	fftw_free(nabnab_dummyTerm_n0v1phi_y);
	fftw_free(nabnab_dummyTerm_n0v1phi_x);
	fftw_free(nabnab_dummyTerm_n1vphi);
	fftw_free(nabnab_dummyTerm_n1vphi_y);
	fftw_free(nabnab_dummyTerm_n1vphi_x);
	fftw_free(nabnab_dummyTerm_n0v0phi1);
	fftw_free(nabnab_dummyTerm_n0v0phi1_y);
	fftw_free(nabnab_dummyTerm_n0v0phi1_x);
	fftw_free(v_nabnab_phiTerm);
	fftw_free(div_dummy_dy0);
	fftw_free(div_dummy_dx0);
	fftw_free(div_dummy_dy1);
	fftw_free(div_dummy_dx1);
	fftw_free(div_dummyTerm);
	fftw_free(div_dummy_y0);
	fftw_free(div_dummy_x0);
	fftw_free(div_dummy_y1);
	fftw_free(div_dummy_x1);
	fftw_free(div_n_nabphi_y);
	fftw_free(div_n_nabphi_x);
	fftw_free(d2Pe1k);
	fftw_free(d2Pi1k);
	fftw_free(ne1k);
	fftw_free(dndx1k);
	fftw_free(dndy1k);
	fftw_free(dphidxk);
	fftw_free(dphidyk);
	fftw_free(vexbkx1);
	fftw_free(vexbky1);
	// fftw_free(phi1k);
	// fftw_free(phi0k);
	fftw_free(coeff);
	fftw_free(coeff1);
	fftw_free(uxBTerm);
	fftw_free(d2phidxk);
	fftw_free(d2phidyk);
	fftw_free(d2phidxyk);
	fftw_free(d2phidyxk);
	fftw_free(d2phidxy1k);
	fftw_free(d2phidyx1k);
	// test
	// fftw_free(potSource_inertia); // REMOVE LATER
	fftw_free(nabnab_dummyTerm_n0v0phi1_xy);
	fftw_free(nabnab_dummyTerm_n0v0phi1_yx);
	fftw_free(nabnab_dummyTerm_n1vphi_xy);
	fftw_free(nabnab_dummyTerm_n1vphi_yx);
	fftw_free(nabnab_dummyTerm_n0v1phi_xy);
	fftw_free(nabnab_dummyTerm_n0v1phi_yx);
	fftw_free(dnabnab_dummyTerm_n0v0phi1_xy);
	fftw_free(dnabnab_dummyTerm_n0v0phi1_yx);
	fftw_free(dnabnab_dummyTerm_n1vphi_xy);
	fftw_free(dnabnab_dummyTerm_n1vphi_yx);
	fftw_free(dnabnab_dummyTerm_n0v1phi_xy);
	fftw_free(dnabnab_dummyTerm_n0v1phi_yx);
	fftw_free(nabnab_dummyTerm_n0v0phi1_yy);
	fftw_free(nabnab_dummyTerm_n1vphi_yy);
	fftw_free(nabnab_dummyTerm_n0v1phi_yy);
}
// ddt_potentialk.m from MATLAB calculates the derivative of potential with time using iterative spectral methods
// double potSourcek[][ncomp] here is just a name it should be potsourcek_inertial in the main code!
// int phi_iter_max is error save in MATLAB: ninvksqu vs invnk

int ddt_potentialk(double invnk[][ncomp], double ninvksqu[], double dndxk[][ncomp], double dndyk[][ncomp], double dphikdt_old[][ncomp], double potSourcek[][ncomp], double kx[], double ky[], double err_max, int max_iter, double dphikdt[][ncomp])
{
	// int ddt_potentialk(double invnk[][ncomp],double ninvksqu[], double dndxk[][ncomp], double dndyk[][ncomp], double dphikdt_old[][ncomp],double potSourcek[][ncomp], double kx[], double ky[], double err_max,  int max_iter, int phi_iter_max, double dphikdt[][ncomp] ){

	// Preallocate grad_dphikdt_old w/ zeros
	fftw_complex *grad_dphikdt_old; // x an y components?
	grad_dphikdt_old = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *grad_dphikdt_old_x; // x an y components?
	grad_dphikdt_old_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *grad_dphikdt_old_y; // x an y components?
	grad_dphikdt_old_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

// init to zeros
#pragma omp parallel for schedule(static) collapse(3)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				grad_dphikdt_old_x[j + nyk * i][k] = 0.;
				grad_dphikdt_old_y[j + nyk * i][k] = 0.;
				grad_dphikdt_old[j + nyk * i][k] = 0.; // or sum of two terms above
			}
		}
	}

	// compute derivatives of above terms wrt kx, ky
	fftw_complex *dgrad_dphikdt_old_x;
	dgrad_dphikdt_old_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *dgrad_dphikdt_old_y;
	dgrad_dphikdt_old_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *gradNgradPhi_x;
	gradNgradPhi_x = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *gradNgradPhi_y;
	gradNgradPhi_y = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *RHS;
	RHS = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// test for printing results
	/*
	double *phi; //= (double**)malloc(sizeof (double*)) * (i-1)); //init here for test
	phi = (double*) fftw_malloc(nx*ny*sizeof(double));
	fftw_complex *phik;
	phik = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));
	*/

	double dphidtkmax, dphikdtmax_old; // phik_max, phik_max_old; //dphikdtmax_old
	double it_error;

	// Begin counter for the number of iterations it takes to calculate phi
	int count = 0;

	do
	{
		// Calculate phi derivatives: Calculate derivatives of dphikdt_old
		// dphikdt_old

		derivk(dphikdt_old, kx, grad_dphikdt_old_x);
		derivk(dphikdt_old, ky, grad_dphikdt_old_y);

		// derivk(grad_dphikdt_old, kx, dgrad_dphikdt_old_x);
		// derivk(grad_dphikdt_old, ky, dgrad_dphikdt_old_y);

		// Do convolutions for grad n dot grad phi term
		// gradNgradphi = [ikx phik] * [ikx nk] - [iky phik] * [iky nk], where * is the convolution
		convolve2D(dndxk, grad_dphikdt_old_x, gradNgradPhi_x);
		convolve2D(dndyk, grad_dphikdt_old_y, gradNgradPhi_y);

// Subtract gradNgradphi from the source term. Calculate RHS of the equation:
#pragma omp parallel for schedule(static) collapse(3)
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
		convolve2D(RHS, invnk, RHS); // dphidtk = - RHS ./ ksqu_4inv;
		//     dphikdtmax_old = max(max(abs(dphidtk_old)));

		// Calculate the old maximum value of dphidtk
		dphikdtmax_old = max_absComp(dphikdt_old);

		Arr3DArr2DMult(RHS, ninvksqu, dphikdt); // dphidtk = - RHS ./ ksqu_4inv;
		// here overwrite old variable with new
		//   Get the new maximum value of the potential
		dphidtkmax = max_absComp(dphikdt);
		// Multiply by ninvksqu to get the updated phi: this will output potential in Fourier space, phik

		// test: Set the old time derivative
		// dphikdt_old = dphikdt;

		// dphidtk is the output we're trying to calculate
		//  % Get the new maximum value of the potential: dphidtkmax = max(max(abs(dphidtk)));
		//  Increase iteration count by 1
		count = count + 1;

		// Calculate error: dphidtkmax
		//         it_error = abs( dphidtkmax - dphikdtmax_old) / dphidtkmax;

		it_error = fabs((dphidtkmax - dphikdtmax_old) / dphidtkmax); // err_max is the error we want to converge to

		// If error is too high and we haven't reached the max iterations yet, repeat iterations
	} while (it_error > err_max && count <= max_iter && dphidtkmax > err_max); // > to keep going// and instead &&
	// test the new phi term here
	// c2rfft(phik,phi);
	// char Phi[] = "Phi_PotSource.txt";
	// print2DArrf(Phi, phi);
	// free(phi);
	// fftw_free(phik);

	fftw_free(grad_dphikdt_old);
	fftw_free(grad_dphikdt_old_x);
	fftw_free(grad_dphikdt_old_y);
	fftw_free(dgrad_dphikdt_old_x);
	fftw_free(dgrad_dphikdt_old_y);

	fftw_free(gradNgradPhi_x);
	fftw_free(gradNgradPhi_y);
	fftw_free(RHS);
	// Output the number of iterations taken
	return count;
}

// Solve for the electric potential using an iterative solver.
//*********************************** inertial TEST potential solver **************************************
//********************************************************************************************************
// make this one function, only change potsourcek to potsourcek_intertial from main code if needed

// ***********************************************************
//************************************************************
//**************************************************************

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
#pragma omp parallel for schedule(static) collapse(2)
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
	} while (it_error > err_max && count <= max_iter && phik_max > err_max); // and instead &&

	fftw_free(dphidxk);
	fftw_free(dphidyk);
	fftw_free(gradNgradPhi_x);
	fftw_free(gradNgradPhi_y);
	fftw_free(RHS);
	// Output the number of iterations taken
	return count;
}

// Calculate the ExB drift
void calcV_ExBk(double dphidxk[][ncomp], double dphidyk[][ncomp], double B[], double B2, double vexbkx[][ncomp], double vexbky[][ncomp])
{ // Your inputs should be dphidxk, dphidyk, B, and B2
#pragma omp parallel for schedule(static) collapse(3)
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

void calc_diamag(double dpdxk[][ncomp], double dpdyk[][ncomp], double B[], double B2, double qa, double nak[][ncomp], double diamagxk[][ncomp], double diamagyk[][ncomp])
{
	cout << "calc_diamag start, thread: [" << omp_get_thread_num() << "]" << endl;
	double start_time = omp_get_wtime();

	fftw_complex *predivx;
	predivx = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *predivy;
	predivy = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

#pragma omp parallel for schedule(static) collapse(3)
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

	double end_time = omp_get_wtime();
	cout << "thread[" << omp_get_thread_num() << "] time taken for calc_diamg in secs: " << end_time - start_time << endl;
}

double calc_dt(double U[], double vexbx[], double vexby[], double diamagxi[], double diamagyi[], double diamagxe[], double diamagye[], double cfl, double kmax, double maxdt)
{

	// double bar = absolute(U);
	double *absArr;
	absArr = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(absArr, 42, nx * ny * sizeof(double));
	/*
	absArr = (double*) fftw_malloc(nx*nyk *sizeof(double));
	memset(absArr, 42, nx*nyk* sizeof(double));
	*/
	//	int N = 3;
	for (int i = 0; i < size; i++)
	{
		if (U[i] < 0)
		{
			absArr[i] = abs(U[i]);
		}
	}
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

void RK4(double f[][ncomp], double dt, double residual[][ncomp], double source[][ncomp], int stage, double fout[][ncomp])
{
	double alpha[4] = {1. / 4, 1. / 3, 1. / 2, 1.};

#pragma omp parallel for schedule(static) collapse(3)
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

#pragma omp parallel for schedule(static) collapse(3)
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

/*void calc_sourcete(double tek[][nyk][ncomp], double tik[][nyk][ncomp], double nk[][nyk][ncomp], double veik[][nyk][ncomp], double me, double mi, double kx[][nyk], double ky[][nyk], double ke, double sourcetek[][nyk][ncomp]){

	double dtexk[nx][nyk][ncomp], dteyk[nx][nyk][ncomp], kedtexk[nx][nyk][ncomp], kedteyk[nx][nyk][ncomp], divkedtek[nx][nyk][ncomp], tetisubk[nx][nyk][ncomp];

	derivk(tek, kx, dtexk);
	derivk(tek, ky, dteyk);

	rscalArr3DMult(dtexk, ke, kedtexk);
	rscalArr3DMult(dteyk, ke, kedteyk);

	Arr3DArr3DAdd(kedtexk, kedteyk, divkedtek);

	Arr3DArr3DSub(tek, tik, tetisubk);






	for (int i = 0; i < nx; i++){
		for (int j = 0; j < nyk; j++){
			for (int k = 0; k < ncomp; k++){
				sourcetek[i][j][k] = ((-1)*(me/mi)*tetisubk[i][j][k]) - ((2/3)*(1/nk[i][j][k])* divkedtek[i][j][k]);
			}
		}
	}
} */

/*void calc_sourceti(double sourcetik[nx][nyk][ncomp], double kx[][nyk], double ky[][nyk], double kb, double ki, double uin, double vin, double vioxk[][nyk][ncomp], double vioyk [][nyk][ncomp], double veoxk[][nyk][ncomp], double veoyk[][nyk][ncomp], double vie, double uie, double nek[][nyk][ncomp], double tik [][nyk][ncomp], double tek[][nyk][ncomp], double tnk [][nyk][ncomp], double me, double mn){

	double dtixk[nx][nyk][ncomp], dtiyk[nx][nyk][ncomp], kidtixk[nx][nyk][ncomp], kidtiyk[nx][nyk][ncomp], divkidtik[nx][nyk][ncomp], vioxsqk[nx][nyk][ncomp], vioysqk[nx][nyk][ncomp], viosqk[nx][nyk][ncomp];
	double vosubxk[nx][nyk][ncomp], vosubyk[nx][nyk][ncomp], vosubxsqk[nx][nyk][ncomp], vosubysqk[nx][nyk][ncomp], vosubsqk[nx][nyk][ncomp], mnviosqk[nx][nyk][ncomp], mevosubsqk[nx][nyk][ncomp];
	double tnsubtik [nx][nyk][ncomp], tesubtik[nx][nyk][ncomp];
	derivk(tik, kx, dtixk);
	derivk(tik, ky, dtiyk);

	rscalArr3DMult(dtixk, ki, kidtixk);
	rscalArr3DMult(dtiyk, ki, kidtiyk);

	Arr3DArr3DAdd(kidtixk, kidtiyk, divkidtik);

	Arr3DArr3DMult(vioxk, vioxk, vioxsqk);
	Arr3DArr3DMult(vioyk, vioyk, vioysqk);
	Arr3DArr3DAdd(vioxsqk, vioysqk, viosqk);

	Arr3DArr3DSub(vioxk, veoxk, vosubxk);
	Arr3DArr3DSub(vioyk, veoyk, vosubyk);
	Arr3DArr3DMult(vosubxk, vosubxk, vosubxsqk);
	Arr3DArr3DMult(vosubyk, vosubyk, vosubysqk);
	Arr3DArr3DAdd(vosubxsqk, vosubysqk, vosubsqk);

	rscalArr3DMult(viosqk, mn, mnviosqk);
	rscalArr3DMult(vosubsqk, me, mevosubsqk);

	Arr3DArr3DSub(tnk, tik, tnsubtik);
	Arr3DArr3DSub(tek, tik, tesubtik);

	for (int i = 0; i < nx; i++){
		for (int j = 0; j < nyk; j++){
			for (int k = 0; k < ncomp; k++){
				sourcetik[i][j][k] = ((2/(3*kb))*((uin*vin)*((3*kb*(tnk[i][j][k] - tik[i][j][k])) + (mn*viosqk[i][j][k]))) + ((uie*vie) * ((3*kb*(tek[i][j][k] - tik[i][j][k])) + mevosubsqk[i][j][k]))) + ((2/(3*nek[i][j][k])) * divkidtik[i][j][k]);
			}
		}
	}
} */
// double ksqu[16] = {0}; //added this
void calc_sourcen(double ksqu[], double nk[][ncomp], double d, double sourcenk[][ncomp])
{

	fftw_complex *lapnk;
	lapnk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	laplaciank(nk, ksqu, lapnk);

	rscalArr3DMult(lapnk, d, sourcenk);

	fftw_free(lapnk); // test
}
