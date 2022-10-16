#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include <math.h>
#include <stdio.h>
#include "spectralFunctions.h"
#include "arithmeticFunctions.h"
#include "fftw3.h"
#include <cstring>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <iostream>
#include <vector>
#include <fstream> //for file operations
#include <iomanip>
#include <omp.h>

// g++ wrappingScript_v2.cpp spectralFunctions.cpp arithmeticFunctions.cpp -lfftw3 -lm -g -o ../../Datasim1_copy/test.out && valgrind --track-origins=yes ./../../Datasim1_copy/test.out
// g++ wrappingScript_v2.cpp spectralFunctions.cpp arithmeticFunctions.cpp -lfftw3 -lm -g -o ../../Datasim1_copy/test.out
//  g++ wrappingScript_v2.cpp spectralFunctions.cpp arithmeticFunctions.cpp -lfftw3 -lm -g -o ../../Datasim1_copy/test.out && valgrind --leak-check=yes ./../../Datasim1_copy/test.out
//   g++ wrappingScript_v2.cpp spectralFunctions.cpp arithmeticFunctions.cpp -lfftw3 -lm -O3 -o  ../../Datasim1_copy/test.out  // run code faster (not for debug)
//-----------------------------------------------------------

// Use the following to run code
// g++ GDI_SLAB.cpp spectralFunctions.cpp arithmeticFunctions.cpp -lfftw3 -lm -O3 -o  ../../Datasim1/test.out

#define size 3 // for the cross product function call
using namespace std;
int main()
{

	// Set script parameters that we can change
	double saveFrequency = 20; // 100.; // 20; //1.0;    // Save data every this many time steps
	double dt_max = 20.;	   // Set max allowable time step
	double tend = 4000.;	   // Set the end time
	double err_max = 1.e-8;	   // Set max allowable error for potential solver
	double CFL = 1.;		   // 3.;             // Set CFL number.
	double Dart = 1.e3;		   // m^2/s // Set artifical diffusion constants

	int phi_iter_max = 500; // Max allowable iterations for potential solver
	int phi_iter = 0;
	int saveNum = 1;
	// test for time step
	// double dt_min = 1.e-8;

	// Calculated parameters
	// int iter_max = tend / dt_max; // Max number of iterations
	int iter_max = 50; // 100000;
	// std::vector<double> uxB = cross(u,B);
	//  NEW set of IC and background conditions

	// Set physical constants
	double e = 1.6021766208E-19;   // e = 1.6021766208e-19;      % C
	double kb = 1.380649E-23;	   // kb = 1.380649e-23;
	double me = 9.10938356E-31;	   // me = 9.10938356e-31;       % kg
	double eps0 = 8.854187817E-12; // eps0 = 8.854187817e-12;    % F/m
	// double nn = 1.E14;   // Neutral particle number density
	double mi = 16. * 1.66053906660E-27; // Mass O+
	// 16 * 1.66053906660e-27;
	double mn = mi;		  // Mass O
	double ri = 152.E-12; // Effective collision radius
	double rn = ri;

	// Set magnetic field. Also calculate different properties of it
	double B[3] = {0, 0, 5E-5};

	// Set neutral wind. Generally, the last element will be 0.
	double u[3] = {-500, 0, 0}; // in KHI neutral wind is 0
	// double u[3] = {-500, 300, 0}; // from MATLAB potentialSolver_test_rigorous
	// double u[3] = {-500, 300, 0}; //ux=u[0]; uy=[1]

	//**************TEST*********************
	double uxB[size];

	double Bmag = mag1DArray(B);
	double B2 = Bmag * Bmag;
	// you can also print this and check it's working
	// cout << "Cross product:";
	cross_product(u, B, uxB);
	// for (int i = 0; i < size; i++)
	// cout << uxB[i] << " ";
	// return 0;

	//	print2DPhysArray(uxB);

	double Oci = e * Bmag / mi;
	double Oce = e * Bmag / me;
	double lgN_right = 3e4;
	;					   // 3e4; multiply 30  by 1e3 to convert to km
	double lgN_left = 3e4; // 3e4;
	// *************************  TEST for the inertial code: **************************************
	// double Cm = (1 / Oci + 1/Oce)^-1; //taking the inverse is not apropriate operator for double so change the ^-1
	double Cm = 1. / (1 / Oci + 1 / Oce);
	double m = 1.;

	// Set domain size parameters.
	double L = 2 * M_PI; // useless
	double Lx = 5.0E5;	 // 8.5E5; //5.0E5; //8.5E5;
	double Ly = 2.5E5;	 // 2.5E5;//10000.; //different
	// double Lx = L, Ly = L;
	double dx = Lx / nx;
	double dy = Ly / ny;

	// I.Cs:  Parameters for initializing hyperbolic tangent ICs
	// for older setup but similiar:
	// equiv to xg ..
	// double xg = Lx * (19./24); //xg center of hyperbolic tangent func
	double xgN_left = 1.25E5;  // 0.75E5 ;//1.25E2;//0.75E5;
	double xgN_right = 3.75E5; // 2.2E5;

	double o = 2.5;			 // 1.; //outer region
	double a = (o - m) / 2.; // aR from chirga's code (thesis)

	// double c = -xg;
	// double d = (o + m)/2.; //size of the cloud ??
	double d = o; // 1.6; //o;
	// a,d,c, a2, c2,d2 are the coefficients for tanh
	double a2 = -1 * a; //-0.3; //for test//-1*a; //aL
	// double c2 = - Lx - c;
	double d2 = d - m;

	// double b = abs(((((tanh(log(m/o)/4)) * a) + d) * pow(cosh(log(m/o)/4), 2))/(a * lg)); //b=1/Lg(1) from MATLAB
	// double bg = 2./lg/lg;

	//***************inertial TEST: add the hyperbolic tan coeffs**********************
	// double ux = u[0]; and double uy= u[1];
	// CFL = 0.9; ???
	double nuin = 0.2;
	double nuen = 1;
	double nn = 1.e12; // 1e14; // 1e8; for high collions and 1e8 for low
	// add this for KHI:
	double V0 = -1000.;
	double xgV_right = 3.75E5; // 3e4;
	double xgV_left = 1.25E5;  // 3e4;

	double lgV_right = 1e4; // 10.;//same LgV = Lg_\phi
	double lgV_left = 1e4;	// 10.; convert km

	double n0 = 1e11;
	double Ampl_1 = -1E-3; // min
	double Ampl_2 = 1E-3;  // max
	// SET Coeff.
	double axphi = 2;
	double ayphi = 1;
	double axyphi = .8;
	double nxphi = 1;	// 2;
	double nyphi = 1;	// 10;
	double nxy1phi = 1; // 2;
	double nxy2phi = 1; // 8;
	double phi0 = 1000 * 5e-5;

	// BG = unperturped, and 0 = perturbed
	double axnBG = 2;
	double aynBG = 4;
	double axynBG = 1;
	double nxnBG = 1;	// 5;
	double nynBG = 1;	// 6;
	double nxy1nBG = 1; // 2;
	double nxy2nBG = 1; // 4;
	double a0nBG = 10;
	double n0BG = 3e11;

	double axphiBG = 2.7;
	double ayphiBG = 0.1;
	double axyphiBG = 1;
	double nxphiBG = 1;
	double nyphiBG = 1;	  // 12;
	double nxy1phiBG = 1; // 6;
	double nxy2phiBG = 1; // 7;
	double phi0BG = 2000 * 5e-5;

	double axdphi = 1;
	double aydphi = 3;
	double axydphi = 2;
	double nxdphi = 1; // 3;
	double nydphi = 1; // 13;
	double nxy1dphi = 1;
	double nxy2dphi = 1;		// 5;
	double dphi0 = 2000 * 5e-5; // perturbed is 0

	// test: for the Mathematica source term
	// double TiBG = 0;
	// double TeBG = 0;
	// double Ti0 = 1000; //same as Ti but distinguish from mathematica soucre term
	// double Te0 = 1000;//2000;

	// Test init real part of potsource inertia
	// double *potSource_inertia; //init here for test
	// potSource_inertia = (double*) fftw_malloc(nx*ny*sizeof(double));
	// memset(potSource_inertia, 42, nx*ny* sizeof(double));

	// Initialize physical grid parameters.
	// XX changes as you change zeroth index. YY changes as you change first index.
	double *XX;
	XX = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(XX, 42, nx * ny * sizeof(double)); // test: XX for some reason isn't the correct dimensions, maybe this is why

	double *YY;
	YY = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(YY, 42, nx * ny * sizeof(double)); // test: XX for some reason isn't the correct dimensions, maybe this is why

	// Make the physical grid
	double kmax = makeSpatialMesh2D(dx, dy, XX, YY);
	// Test for timestep
	// double Dmax = std::max(Dart,Dart);
	double dt_Dmax = CFL / (Dart * pow(kmax, 2)); // time step associated with D max
	// makeFourierMesh2D

	// if (dt_max < dt_Dmax){
	// return dt_max;
	// }else {
	//	return dt_Dmax;
	// }
	// if (dt_max>dt_Dmax)dt_max=dt_Dmax;
	dt_max = std::min<double>(dt_max, dt_Dmax);
	// std::min takes more than 1 argument using {} so from MATLAB this here will be:
	// dt_max = min({dt_max,dt_Dmax});
	// std::min(dt_max,dt_Dmax);

	// print2DPhysArray(XX);
	//  Initialize Fourier grid parameters.

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

	int FourierMeshType = 1; // 1;// 1; // this "flag" should not be double, this should generally be avoided so make it int and compare it against ints instead
	// Make the Fourier grid
	makeFourierMesh2D(Lx, Ly, kx, ky, ksqu, ninvksqu, FourierMeshType);
	// Make the Fourier grid
	// makeFourierMesh2D(Lx, Ly, kx, ky, ksqu, ninvksqu);

	// Initialize variables to evolve
	double *ne;
	ne = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(ne, 42, nx*nyk* sizeof(double)); //test extra to testbed

	fftw_complex *nek;
	nek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// memset(nek, 42, nx*nyk* sizeof(fftw_complex)); //test extra to testbed

	//**************************TEST for inertial*******************************
	//***************************************************************************
	double *ne1;
	ne1 = (double *)fftw_malloc(nx * ny * sizeof(double)); // initialized in for loop. Used free for a test

	// initialize all background to an array or vector of zeros
	double *ne0;
	ne0 = (double *)fftw_malloc(nx * ny * sizeof(double)); // initialized in for loop. Used free for a test
	// std::vector<double> ne0(nx*ny ,0.0); // you could try using memset as well

	fftw_complex *ne0k; // for density perturbations
	ne0k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(ne0k, 42, nx * nyk * sizeof(fftw_complex)); // test extra to testbed

	fftw_complex *ne1k; // for density perturbations
	ne1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(ne1k, 42, nx * nyk * sizeof(fftw_complex)); // test extra to testbed

	//**********************************************

	double *Ti;
	Ti = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(Ti, 42, nx*nyk* sizeof(double)); //test extra to testbed //Ti is initialized in for loop
	//*****************inertial TEST*********
	double *Ti_0;
	Ti_0 = (double *)fftw_malloc(nx * ny * sizeof(double)); // Ti_1
	// memset(Ti0, 42, nx*nyk* sizeof(double)); //from mathematica note. TBG = 0 so initialize to 0?
	// std::vector<double> Ti_0(nx*ny ,0.0); // conflict in declarization
	double *Ti_1;
	Ti_1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	double *Te_1;
	Te_1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	fftw_complex *Tik;
	Tik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Tik, 42, nx * nyk * sizeof(fftw_complex)); // test extra to testbed

	fftw_complex *Ti0k;
	Ti0k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Ti0k, 42, nx * nyk * sizeof(fftw_complex));

	double *Te;
	Te = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(Te, 42, nx*nyk* sizeof(double)); //test extra to testbed //this with ne is initialized in for loop
	//*****************inertial TEST***********
	double *Te_0;
	Te_0 = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(Te0, 42, nx*nyk* sizeof(double)); //from mathematica note. TBG = 0 so initialize to 0?
	// std::vector<double> Te_0(nx*ny ,0.0); // you could try using memset as well

	fftw_complex *Tek;
	Tek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Tek, 42, nx * nyk * sizeof(fftw_complex)); // test extra to testbed

	fftw_complex *Te0k;
	Te0k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Te0k, 42, nx * nyk * sizeof(fftw_complex)); // test extra to testbed

	double *phi;
	phi = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(phi, 42, nx*nyk* sizeof(double)); //test extra to testbed // initialized in for loop?

	double *phi1;
	phi1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(phi, 42, nx*nyk* sizeof(double)); //test extra to testbed // initialized in for loop?

	double *phi_0;
	phi_0 = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(phi, 42, nx*nyk* sizeof(double)); //test extra to testbed // initialized in for loop?
	// std::vector<double> phi_0(nx*ny ,0.0); // you could try memset as well

	fftw_complex *phik;
	phik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(phik, 42, nx * nyk * sizeof(fftw_complex)); // test

	// *************** Inertial TEST********************
	fftw_complex *phi1k;
	phi1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(phi1k, 42, nx * nyk * sizeof(fftw_complex)); // do not know if this memset is needed

	fftw_complex *phi0k;
	phi0k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(phi0k, 42, nx * nyk * sizeof(fftw_complex));

	//***************************************************
	// Initialize pressures
	double *Pi;
	Pi = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(Pi, 42, nx*nyk* sizeof(double)); //test extra to testbed, Pi and Pe are initialized in a foor loop down

	fftw_complex *Pik;
	Pik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Pik, 42, nx * nyk * sizeof(fftw_complex)); // test

	//**************************TEST for inertial*******************************
	//***************************************************************************
	fftw_complex *Pi1k; // for perturbation
	Pi1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Pi1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *Pi0k; // for perturbation
	Pi0k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Pi0k, 42, nx * nyk * sizeof(fftw_complex));

	// initialize Pi0 as well

	double *Pi0;
	Pi0 = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(Pi0, 42, nx*nyk* sizeof(double)); //initialized in the loop

	double *Pi1;
	Pi1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(Pi0, 42, nx*nyk* sizeof(double)); //initialized in the loop

	//*********************************************************
	double *Pe;
	Pe = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(Pe, 42, nx * nyk * sizeof(double)); // test extra to testbed Pe and Pi are initialized in a loop down

	fftw_complex *Pek;
	Pek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Pek, 42, nx * nyk * sizeof(fftw_complex)); // test

	//**************************TEST for inertial*******************************
	//***************************************************************************
	fftw_complex *Pe1k; // for perturbation
	Pe1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Pe1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *Pe0k; // for perturbation
	Pe0k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(Pe0k, 42, nx * nyk * sizeof(fftw_complex));

	// initialize pe0 as well
	double *Pe0;
	Pe0 = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(Pe0, 42, nx*nyk* sizeof(double)); //initialized in the loop

	double *Pe1;
	Pe1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	// memset(Pe0, 42, nx*nyk* sizeof(double)); //initialized in the loop

	//********************************************
	// Initialize velocities

	double *vexbx;
	vexbx = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vexbx, 42, nx * nyk * sizeof(double)); // added this here extra to testbed

	double *vexbx0;
	vexbx0 = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vexbx0, 42, nx * nyk * sizeof(double));

	double *vexbx1;
	vexbx1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vexbx1, 42, nx * nyk * sizeof(double));

	fftw_complex *vexbkx; // change name from vexbxk to vexbkx (defined differently in spectral funcs)
	vexbkx = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vexbkx, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	//******************************Inertial TEST**********************************************
	//*****************************************************************************************

	fftw_complex *vexbkx0; // change name from vexbxk to vexbkx (defined differently in spectral funcs)
	vexbkx0 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vexbkx0, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *vexbkx1; // change name from vexbxk to vexbkx (defined differently in spectral funcs)
	vexbkx1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vexbkx1, 42, nx * nyk * sizeof(fftw_complex)); // veox1k

	double *vexby;
	vexby = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vexby, 42, nx * nyk * sizeof(double)); // added this here extra to testbed since calc dt not tested

	double *vexby0;
	vexby0 = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vexby0, 42, nx * nyk * sizeof(double));

	double *vexby1;
	vexby1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vexby1, 42, nx * nyk * sizeof(double));

	fftw_complex *vexbky; // change name from vexbyk to vexbky (defined differently in spectral funcs)
	vexbky = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vexbky, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	//******************************Inertial TEST**********************************************
	//*****************************************************************************************

	fftw_complex *vexbky0; // change name from vexbxk to vexbkx (defined differently in spectral funcs)
	vexbky0 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vexbky0, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *vexbky1; // change name from vexbxk to vexbkx (defined differently in spectral funcs)
	vexbky1 = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vexbky1, 42, nx * nyk * sizeof(fftw_complex));

	//*****************************************************
	double *vdmex;
	vdmex = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmex, 42, nx * ny * sizeof(double)); // added this here extra to testbed since calc dt not tested
	// TEST
	double *vdmex1;
	vdmex1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmex1, 42, nx * ny * sizeof(double));

	fftw_complex *vdmexk;
	vdmexk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmexk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed
	// TEST FOR PERTURBED
	fftw_complex *vdmex1k;
	vdmex1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmex1k, 42, nx * nyk * sizeof(fftw_complex));

	double *vdmey;
	vdmey = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmey, 42, nx * ny * sizeof(double)); // added this here extra to testbed since calc dt not tested
	// TEST
	double *vdmey1;
	vdmey1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmey1, 42, nx * ny * sizeof(double));

	fftw_complex *vdmeyk;
	vdmeyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmeyk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed
	// TEST FOR PERTURBED
	fftw_complex *vdmey1k;
	vdmey1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmey1k, 42, nx * nyk * sizeof(fftw_complex));

	double *vdmix;
	vdmix = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmix, 42, nx * ny * sizeof(double)); // added this here extra to testbed since calc dt not tested
	double *vdmix1;
	vdmix1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmix1, 42, nx * ny * sizeof(double));

	fftw_complex *vdmixk;
	vdmixk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmixk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed
	// TEST FOR PERTURBED
	fftw_complex *vdmix1k;
	vdmix1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmix1k, 42, nx * nyk * sizeof(fftw_complex));

	double *vdmiy;
	vdmiy = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmiy, 42, nx * ny * sizeof(double)); // added this here extra to testbed since calc dt not tested
	// TEST PERTURBED //init vdmex1 vdmey1 vdmix1 vdmiy1
	double *vdmiy1;
	vdmiy1 = (double *)fftw_malloc(nx * ny * sizeof(double));
	memset(vdmiy1, 42, nx * ny * sizeof(double));

	fftw_complex *vdmiyk;
	vdmiyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmiyk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed
	// TEST FOR PERTURBED
	fftw_complex *vdmiy1k;
	vdmiy1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vdmiy1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *veoxk; // veox1k
	veoxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(veoxk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed
	// TEST PERTURBED
	fftw_complex *veox1k;
	veox1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(veox1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *veoyk;
	veoyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(veoyk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed
	// TEST PERTURBED
	// fftw_complex *veoy1k; //not used for now
	// veoy1k = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));
	// memset(veoy1k, 42, nx*nyk* sizeof(fftw_complex));

	fftw_complex *vioxk;
	vioxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vioxk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed
	// TEST PERTURBED
	// fftw_complex *viox1k; // not used for now
	// viox1k = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));
	// memset(viox1k, 42, nx*nyk* sizeof(fftw_complex));

	fftw_complex *vioyk;
	vioyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(vioyk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed
	// TEST PERTURBED
	// fftw_complex *vioy1k; //not used for now
	// vioy1k = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));
	// memset(vioy1k, 42, nx*nyk* sizeof(fftw_complex));

	//**************************TEST for inertial*******************************
	//***************************************************************************

	double start_time = omp_get_wtime();

	// Residuals and source terms

	fftw_complex *residualnk;
	residualnk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(residualnk, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed residualn1k

	// add this for ne1k
	// fftw_complex *residualn1k; // not used for now and residualnk is enough substition
	// residualn1k = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));
	// memset(residualn1k, 42, nx*nyk* sizeof(fftw_complex));
	// ADD phi1k residualk_phi
	fftw_complex *residualk_phi;
	residualk_phi = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// memset(residualk_phi, 42, nx*nyk* sizeof(fftw_complex));
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				residualk_phi[j + nyk * i][k] = 0;
			}
		}
	}

	fftw_complex *residualtik;
	residualtik = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(residualtik, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	fftw_complex *residualtek;
	residualtek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(residualtek, 42, nx * nyk * sizeof(fftw_complex)); // added this here extra to testbed

	// fftw_complex *sourcenk; //not used for now
	// sourcenk = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));
	// memset(sourcenk, 42, nx*nyk* sizeof(fftw_complex)); //added this here extra to testbed
	// add this for the pert term sourcen1k
	fftw_complex *sourcen1k;
	sourcen1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(sourcen1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *sourcetk;
	sourcetk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex)); // this is initialized by for loop
	// add as test init to zeros maybe: sourcenk, sourcen1k
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				sourcetk[j + nyk * i][k] = 0;
			}
		}
	}
	// old thing for perturbed variables
	fftw_complex *nek_old; // not used here in inertial part so
	nek_old = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// memset(nek_old, 42, nx*nyk* sizeof(fftw_complex)); //added this here extra to testbed //initialized in for loop
	fftw_complex *ne1k_old; // old for perturbed variables only
	ne1k_old = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

	fftw_complex *Tik_old; // this is pert Ti
	Tik_old = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// memset(Tik_old, 42, nx*nyk* sizeof(fftw_complex)); //added this here extra to testbed // initialized in for loop
	// fftw_complex *Ti1k_old; // Ti_0 is unperturbed and Ti and Te is pert
	// Ti1k_old = (fftw_complex*) fftw_malloc(nx*nyk* sizeof(fftw_complex));

	fftw_complex *Tek_old; // this is pert Te
	Tek_old = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// memset(Tek_old, 42, nx*nyk* sizeof(fftw_complex)); //added this here extra to testbed initialized in for loop

	// add for potential
	fftw_complex *phi1k_old; // this is pert Te
	phi1k_old = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));

// This will likely change but leave these for now until we get real numbers to use.
// Set constants used to make initial conditions

// double test = sin((3/Ly)*M_PI * YY[8][9]);

// Set initial conditions only for ne, Ti, and Te
// primv variables are p, T, phi,.. > Phi = Phi0+Phi1
// unpert var are p0, n0, phi1
// perturbed are n1, phi1, p1,...
// phi in MATLAB initialized to func "tanhfromSech2IC" and ne, Te, Ti are initizlied to "tanhIC"
// double randArr[nx*ny];
#pragma omp parallel for schedule(static)
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < ny; j++)
		{
			ne1[j + ny * i] = ((rand() % 101) / 100. * (Ampl_2 - Ampl_1) + Ampl_1) * n0;
			// ne0[j + ny*i] = (a2 * tanh((XX[j + ny*i] - xgN_left)/ lgN_left) + a * tanh((XX[j + ny*i] - xgN_right)/lgN_right) - a * tanh((XX[j + ny*i] - Lx + xgN_right)/lgN_right) - a2 * tanh((XX[j + ny*i] - Lx + xgN_left)/lgN_left) + d)*n0;
			ne0[j + ny * i] = (a2 * tanh((XX[j + ny * i] - xgN_left) / lgN_left) + a * tanh((XX[j + ny * i] - xgN_right) / lgN_right) + d) * n0; //- a * tanh((XX[j + ny*i] - Lx + xgN_right)/lgN_right) - a2 * tanh((XX[j + ny*i] - Lx + xgN_left)/lgN_left) + d)*n0;
																																				 // this 2 hyper tanh is for GDI slab
			//(axnBG*sin(2*M_PI *nxnBG * XX[j + ny*i]/Lx) + aynBG*cos(2*M_PI *nynBG * YY[j + ny*i]/Ly) + axynBG*cos(2*M_PI *nxy1nBG * XX[j + ny*i]/Lx) * sin(2*M_PI *nxy2nBG * YY[j + ny*i]/Lx) + a0nBG) * n0BG;
			ne[j + ny * i] = ne0[j + ny * i] + ne1[j + ny * i];
			Ti_1[j + ny * i] = 0.; // Ti = Ti1 pert
			Te_1[j + ny * i] = 0.;
			Ti_0[j + ny * i] = 1000.; // 0.; //TEST like TBG=0, so delete memset for Te0 and Ti0 for test: this causes code to repeate phi iter: 1,7,71,1,7..
			Te_0[j + ny * i] = 1000.; // 2nd TEST switch temps around, TP= 0 and TBG for 1000
			Te[j + ny * i] = Te_1[j + ny * i] + Te_0[j + ny * i];
			Ti[j + ny * i] = Ti_1[j + ny * i] + Ti_0[j + ny * i];

			phi_0[j + ny * i] = 0.; // for GDI SLAB
			// phi_0[j + ny*i] = (B[2]*V0)/2.*(lgV_left* log(cosh((XX[j + ny*i] - xgV_left)/ lgV_left)) - lgV_right* log(cosh((XX[j + ny*i] - xgV_right)/ lgV_right)) - lgV_right* log(cosh((XX[j + ny*i] - Lx + xgV_right)/ lgV_right)) + lgV_left* log(cosh((XX[j + ny*i] - Lx + xgV_left)/ lgV_left)) ); //change from: phi_0[j + ny*i] = 0.;
			// phi_0[j + ny*i] = (B[2]*V0)*(lgV_left* tanh((XX[j + ny*i] - xgV_left)/ lgV_left) - lgV_right* tanh((XX[j + ny*i] - xgV_right)/ lgV_right)); //- lgV_right* log(cosh((XX[j + ny*i] - Lx + xgV_right)/ lgV_right)) + lgV_left* log(cosh((XX[j + ny*i] - Lx + xgV_left)/ lgV_left)) ); //change from: phi_0[j + ny*i] = 0.;
			phi1[j + ny * i] = 0.;
			phi[j + ny * i] = phi_0[j + ny * i] + phi1[j + ny * i];

			// Calculate pressures here:  primative (total) pressure
			Pi[j + ny * i] = ne[j + ny * i] * Ti[j + ny * i] * kb; // Kb*convolove2D(prim ne, primative Ti) = Pi (no need for convo since its in real space)
			Pe[j + ny * i] = ne[j + ny * i] * Te[j + ny * i] * kb; // Kb*convolove2D(prim ne, prim Te) = Pe
			// Get unperturbed pressures: ideal gas law
			// initialize Pi0 and Pe0 above?
			Pi0[j + ny * i] = ne0[j + ny * i] * Ti_0[j + ny * i] * kb; // TiBG = 0 so initialized to 0? this gives nan after 4th iteration. Try Ti=Ti0, this makes func blows up
			Pe0[j + ny * i] = ne0[j + ny * i] * Te_0[j + ny * i] * kb;
			// Get perturbed pressures:
			Pi1[j + ny * i] = Pi[j + ny * i] - Pi0[j + ny * i];
			Pe1[j + ny * i] = Pe[j + ny * i] - Pe0[j + ny * i];
		}
	}
	// print2DPhysArray(ne);

	double end_time = omp_get_wtime();
	cout << "time taken in secs: " << end_time - start_time << endl;

	double transform_start_time = omp_get_wtime();
	// Convert all of these to Fourier space by taking their Fourier transforms //Take FFT of all variables
	r2cfft(ne, nek);
	r2cfft(Ti, Tik);
	r2cfft(Te, Tek);
	r2cfft(phi, phik);
	r2cfft(Pi, Pik);
	r2cfft(Pe, Pek);
	//************** Inertial TEST ******************
	r2cfft(ne0, ne0k);
	r2cfft(ne1, ne1k);
	// not sure Te0, Ti0 to Te0k and Ti0k?? then add initialization for Te0k and Ti0k
	r2cfft(Te_0, Te0k);
	r2cfft(Ti_0, Ti0k);

	r2cfft(phi_0, phi0k);
	r2cfft(phi1, phi1k);
	r2cfft(Pi0, Pi0k);
	r2cfft(Pi1, Pi1k);
	r2cfft(Pe0, Pe0k);
	r2cfft(Pe1, Pe1k);

	double transform_end_time = omp_get_wtime();
	cout << "time taken for transforms in secs: " << transform_end_time - transform_start_time << endl;

	// uxb term test for the inertial function
	// no need for this here, this is just a constant--vector or scalar and there's no z component
	// maybe initialize these components here to use it as an input for inertia func?
	// For example:
	// test

	// char PotSourceM[] = "PotSourceM.txt";
	// print2DArrf(PotSourceM, Testest);
	// print2DPhysArray(Testest);

	//**************************TEST for inertial*******************************
	//***************************************************************************
	// Calculate ion nd electron gyrofrequencies - qb/m

	// Initialize and calculate collision frequencies

	//**************************TEST for inertial*******************************
	//***************************************************************************
	// should I initialize hall terms in a foor loop here? then I don't have to use memset
	fftw_complex *hallEk;
	hallEk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(hallEk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *hallIk;
	hallIk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(hallIk, 42, nx * nyk * sizeof(fftw_complex)); // test

	//****************************************************************

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

	// double *Testt; //= (double**)malloc(sizeof (double*)) * (i-1)); //init here for test
	// Testt = (double*) fftw_malloc(nx*ny*sizeof(double));

	// char neinitialg[] = "ne_initial.gkyl";
	// print2DB(neinitialg, ne);

	//******************** inertial TEST
	calcCollFreqk_inertia(nek, Tik, Tek, kb, eps0, mi, me, ri, rn, nn, Oci, Oce, e, nuink, nuiek, nuiik, nuenk, nueek, nueik, isigPk, invnk, hallIk, hallEk);

	// chnaged from:
	// calcCollFreqk(nek, Tik, Tek , kb, eps0, mi, me, ri, rn, nn, Oci, Oce, e, nuink, nuiek, nuiik, nuenk, nueek, nueik, isigPk, invnk);
	// Calculate initial potential here

	fftw_complex *potSourcek;
	potSourcek = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// initialize potSourceK, use memset for bits and for loop for values
	memset(potSourcek, 42, nx * nyk * sizeof(fftw_complex)); // test

	//************************inertial TEST ******************************
	//*******************************************************************

	fftw_complex *potSourcek_inertia;
	potSourcek_inertia = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(potSourcek_inertia, 42, nx * nyk * sizeof(fftw_complex));

	// print1DArray(potSourcek_inertia); // test, not sure. wanna see output of this function
	//  can not use print2DPhysArray here cause this is for double [][ny] and potsource is double [][2] so try print1DArray instead
	//  can not use print1DArray cause it's for double[] not double [][2], try printf ?
	//  try: printf("%d\n", potSourcek_inertia)
	// printf("%9.5f \n", potSourcek_inertia); //tested %e
	// printf("(%.15f, %.15f)\n", (*potSourcek_inertia)[0], (*potSourcek_inertia)[1]); //this works with 0000 but in spectral gives two repeated values

	//**************************************************************************
	// *************************** inertial TEST *******************************
	// added this line for velocity calculation test(copied from below)
	// Derivatives

	fftw_complex *dndxk;
	dndxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dndxk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *dndyk;
	dndyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dndyk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *dndx0k; // this is for the density perturbations
	dndx0k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dndx0k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *dndy0k; // this is for the density perturbations
	dndy0k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dndy0k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *dndx1k; // this is for the density perturbations
	dndx1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dndx1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *dndy1k; // this is for the density perturbations
	dndy1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dndy1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *dphidxk;
	dphidxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// memset(dphidxk, 42, nx*nyk* sizeof(fftw_complex)); //phik is defined and this is its derivative wrt kx

	fftw_complex *dphidyk;
	dphidyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// memset(dphidyk, 42, nx*nyk* sizeof(fftw_complex)); //phik is defined and this is its derivative wrt kx

	//**************************TEST for inertial*******************************
	//***************************************************************************

	fftw_complex *dphidx0k; // this is for the density perturbations
	dphidx0k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dphidx0k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *dphidx1k; // this is for the density perturbations
	dphidx1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dphidx1k, 42, nx * nyk * sizeof(fftw_complex));

	//
	fftw_complex *dphidy0k; // this is for the density perturbations
	dphidy0k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dphidy0k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *dphidy1k; // this is for the density perturbations
	dphidy1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dphidy1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *d2phidxk; // initialize this in the main code as well?
	d2phidxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(d2phidxk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *d2phidyk;
	d2phidyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(d2phidyk, 42, nx * nyk * sizeof(fftw_complex)); // test

	fftw_complex *dpedxk;
	dpedxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpedxk, 42, nx * nyk * sizeof(fftw_complex)); // test
	// ADD TEST ONLY PERTURBED TERM
	fftw_complex *dpedx1k;
	dpedx1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpedx1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *dpedyk;
	dpedyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpedyk, 42, nx * nyk * sizeof(fftw_complex)); // test
	// ADD TEST ONLY PERTURBED TERM
	fftw_complex *dpedy1k;
	dpedy1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpedy1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *dpidxk;
	dpidxk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpidxk, 42, nx * nyk * sizeof(fftw_complex)); // test
	// ADD TEST ONLY PERTURBED TERM
	fftw_complex *dpidx1k;
	dpidx1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpidx1k, 42, nx * nyk * sizeof(fftw_complex));

	fftw_complex *dpidyk;
	dpidyk = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpidyk, 42, nx * nyk * sizeof(fftw_complex)); // test
	// ADD TEST ONLY PERTURBED TERM
	fftw_complex *dpidy1k;
	dpidy1k = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	memset(dpidy1k, 42, nx * nyk * sizeof(fftw_complex));

	// define dummy variable for potential solver: dphikdt
	fftw_complex *dphikdt;
	dphikdt = (fftw_complex *)fftw_malloc(nx * nyk * sizeof(fftw_complex));
	// memset(dphikdt, 42, nx*nyk* sizeof(fftw_complex));
	for (int i = 0; i < nx; i++)
	{
		for (int j = 0; j < nyk; j++)
		{
			for (int k = 0; k < ncomp; k++)
			{
				dphikdt[j + nyk * i][k] = 1.;
			}
		}
	}
	// EXTRA TEST:
	// calcPotSourcek_inertia(ne0k, nek, dndx0k, dndy0k, dndxk, dndyk, dphidx0k, dphidy0k, dphidx1k, dphidy1k, Pi1k, Pe1k, uxB, e, Cm, hallEk, hallIk, vexbkx0, vexbky0, vexbkx, vexbky, kx, ky, ksqu, potSourcek_inertia);
	// int phi_iter = potentialk(invnk, dndxk, dndyk, phik, potSourcek_inertia, kx, ky, ninvksqu, err_max, phi_iter_max);

	double derivek_start_time = omp_get_wtime();

	derivk(nek, kx, dndxk);
	derivk(nek, ky, dndyk);
	// add ne0k
	derivk(ne0k, kx, dndx0k);
	derivk(ne0k, ky, dndy0k);
	// add ne1k //dndx1k dndy1k
	derivk(ne1k, kx, dndx1k);
	derivk(ne1k, ky, dndy1k);

	derivk(phik, kx, dphidxk);
	derivk(phik, ky, dphidyk);
	// add phi0k
	derivk(phi0k, kx, dphidx0k);
	derivk(phi0k, ky, dphidy0k);
	// add phi1k
	derivk(phi1k, kx, dphidx1k);
	derivk(phi1k, ky, dphidy1k);

	derivk(Pek, kx, dpedxk);
	derivk(Pek, ky, dpedyk);
	derivk(Pik, kx, dpidxk);
	derivk(Pik, ky, dpidyk);
	// add Pe1k and Pi1k ONLY for now
	derivk(Pe1k, kx, dpedx1k);
	derivk(Pe1k, ky, dpedy1k);
	derivk(Pi1k, kx, dpidx1k);
	derivk(Pi1k, ky, dpidy1k);

	double derivek_end_time = omp_get_wtime();
	cout << "time taken for derivek in secs: " << derivek_end_time - derivek_start_time << endl;

	// calculate all velocities: vexbk, vdmik, vdmek

	// Calculate all velocities

	double calc_start_time = omp_get_wtime();

	calcV_ExBk(dphidxk, dphidyk, B, B2, vexbkx, vexbky);

#pragma omp parallel
	{
#pragma omp single nowait
		c2rfft(vexbkx, vexbx);
#pragma omp single nowait
		c2rfft(vexbky, vexby);
// add dphidx0k, dphidy0k
#pragma omp single nowait
		calcV_ExBk(dphidx0k, dphidy0k, B, B2, vexbkx0, vexbky0);
#pragma omp single nowait
		c2rfft(vexbkx0, vexbx0);
#pragma omp single nowait
		c2rfft(vexbky0, vexby0);
// add dhpidx1k and dphidy1
#pragma omp single nowait
		calcV_ExBk(dphidx1k, dphidy1k, B, B2, vexbkx1, vexbky1); //
#pragma omp single nowait
		c2rfft(vexbkx1, vexbx1);
#pragma omp single nowait
		c2rfft(vexbky1, vexby1);
	}

// maybe add 0k and 1k terms here too?
#pragma omp single nowait
	calc_diamag(dpedxk, dpedyk, B, B2, -1 * e, nek, vdmexk, vdmeyk); ////vdmex1k vdmey1k
#pragma omp single nowait
	calc_diamag(dpidxk, dpidyk, B, B2, e, nek, vdmixk, vdmiyk);

// ADD 0K AND 1K (for now only 1k) // dpedx1k dpedy1k dpidx1k dpidy1k
#pragma omp single nowait
	calc_diamag(dpedx1k, dpedy1k, B, B2, -1 * e, ne1k, vdmex1k, vdmey1k); ////vdmex1k vdmey1k
#pragma omp single nowait
	calc_diamag(dpidx1k, dpidy1k, B, B2, e, ne1k, vdmix1k, vdmiy1k); // vdmiy1k vdmix1k

	cout << "== second level==" << endl;

#pragma omp parallel
	{
#pragma omp single nowait
		c2rfft(vdmexk, vdmex);
#pragma omp single nowait
		c2rfft(vdmeyk, vdmey);
#pragma omp single nowait
		c2rfft(vdmixk, vdmix);
#pragma omp single nowait
		c2rfft(vdmiyk, vdmiy);

		cout << "==2nd level - phase1 done.." << endl;

// add
#pragma omp single nowait
		c2rfft(vdmex1k, vdmex1); // init vdmex1 vdmey1 vdmix1 vdmiy1
#pragma omp single nowait
		c2rfft(vdmey1k, vdmey1);
#pragma omp single nowait
		c2rfft(vdmix1k, vdmix1);
#pragma omp single nowait
		c2rfft(vdmiy1k, vdmiy1);

		cout << "==2nd level - phase2 done.." << endl;

		// calcPotSourcek_inertia(ne0k, nek, dndx0k, dndy0k, dndxk, dndyk, dphidx0k, dphidy0k, dphidx1k, dphidy1k, Pi1k, Pe1k, uxB, e, Cm, hallEk, hallIk, vexbkx0, vexbky0, vexbkx, vexbky, kx, ky, ksqu, potSourcek_inertia);

		// calcPotSourcek(dndxk, dndyk, Pik, Pek, nuink, nuiek, nuenk, nueik, isigPk, Oci, Oce, u, B, ksqu, potSourcek);
		//  commment everything under

		//****************** inertial TEST *************************
		// int phi_iter = potentialk(invnk, dndxk, dndyk, phik, potSourcek_inertia, kx, ky, ninvksqu, err_max, phi_iter_max); // potentialk_inertia is the same as potentialk with different source input
		// int phi_iter = potentialk(invnk, dndxk, dndyk, phik, potSourcek, kx, ky, ninvksqu, err_max, phi_iter_max);

		// Initialize a time vector of length iter_max+1.
		std::vector<double> time(iter_max + 1, 0.0); // This will create a vector of size iter_max + 1 all initialized to 0.0. You could use memset as well
// g++ for correct linker

// double time[iter_max + 1];
// time[0] = 0;
//  Save initial conditions
#pragma omp single nowait
		c2rfft(nek, ne);
#pragma omp single nowait
		c2rfft(Tek, Te);
#pragma omp single nowait
		c2rfft(Tik, Ti);
#pragma omp single nowait
		c2rfft(phik, phi);
		// test: unperturbed?
		// c2rfft(ne0k, ne0);
		// c2rfft(ne1k, ne1);
		// c2rfft(Tek, Te);
	}

	double calc_end_time = omp_get_wtime();
	cout << "time taken for calc in secs: " << calc_end_time - calc_start_time << endl;

	// **********inertial TEST ************
	// save unperturbed

	/*
		//commenting this out as file saving is common to all

		char XXgrid[] = "X.txt";
		// char YYgrid[] = "Y.gkyl";
		char YYgrid[] = "Y.txt";

		print2DArrf(XXgrid, XX);
		// print2DB(XXgrid, XX); //test
		print2DArrf(YYgrid, YY);
		// print2DB(YYgrid, YY); //test

		// output pert n
		// char ne_perturbed[] = "ne_perturbed.gkyl";
		char ne_perturbed1[] = "ne_perturbed.gkyl";
		char ne_perturbed[] = "ne_perturbed.txt";

		print2DArrf(ne_perturbed, ne1);
		print2DB(ne_perturbed1, ne1);

		char ne_unpert1[] = "ne_unpert.gkyl";
		char ne_unpert[] = "ne_unpert.txt";
		print2DArrf(ne_unpert, ne0);
		print2DB(ne_unpert1, ne0);

		char phi_unpert[] = "phi_unpert.txt";
		char phi_unpert1[] = "phi_unpert.gkyl";

		print2DArrf(phi_unpert, phi_0);
		print2DB(phi_unpert1, phi_0);

		char phi_perturbed1[] = "phi_perturbed.gkyl";
		char phi_perturbed[] = "phi_perturbed.txt";

		// print2DB(phi_perturbed, phi1);
		print2DArrf(phi_perturbed, phi1);
		print2DB(phi_perturbed1, phi1);

		*/

	// here for one func with a switch
	// all of the above here can go in one function: inertial vs noninertial switch for this function

	// start here do each function serparately --testing for memory leak issues

	// ***** Comment this out, the RK step and just run or test the potential source separatly and print it out ******
	// ***************************************************************************************************************
	//****************************************************************************************************************
	//****************************************************************************************************************
	// extra TEST: get all velcoities

	// Begin for loop for stepping forward in time
	//	for (int iter = 0; iter < iter_max; iter++){

	/* commenting this out as pointed out in the video
		for (int iter = 0; iter < iter_max; iter++)
		{
			// total veloc
			c2rfft(vexbkx, vexbx);
			c2rfft(vexbky, vexby);
			// diamg e, ion

			c2rfft(vdmexk, vdmex);
			c2rfft(vdmeyk, vdmey);
			c2rfft(vdmixk, vdmix);
			c2rfft(vdmiyk, vdmiy);
			// dt
			double dt = calc_dt(u, vexbx, vexby, vdmix, vdmiy, vdmex, vdmey, CFL, kmax, dt_max);
			// double dt = calc_dt(vexbx, vexby, vdmix, vdmiy, vdmex, vdmiy, CFL, kmax, dt_max); // vdmiy is repeated
			// if time step too small break out foor loop:
			// if (dt < dt_min){
			//	break;
			//}
			// update time TEST

			// Update time
			time[iter + 1] = time[iter] + dt; // break?

	#pragma omp parallel for schedule(static) collapse(3)
			for (int i = 0; i < nx; i++)
			{
				for (int j = 0; j < nyk; j++)
				{
					for (int k = 0; k < ncomp; k++)
					{
						ne1k_old[j + nyk * i][k] = ne1k[j + nyk * i][k]; // change to perturbed all of them: ne1k
						Tek_old[j + nyk * i][k] = Tek[j + nyk * i][k];	 // Tek and Ti are pertrubed
						Tik_old[j + nyk * i][k] = Tik[j + nyk * i][k];
						phi1k_old[j + nyk * i][k] = phi1k[j + nyk * i][k]; // added phi1k
					}
				}
			}

			// double *test;
			// test = (double*) fftw_malloc(nx*ny*sizeof(double));

			// Begin RK method

			for (int stage = 0; stage < 4; stage++)
			{

				calc_residualn(vexbkx, vexbky, nek, residualnk, kx, ky);
				// print

				calc_residualt(vioxk, vioyk, Tik, residualtik, kx, ky); // tik and tek are the pert terms

				// calc_residualt(viox1k, vioy1k, Tik, residualtik, kx, ky);

				calc_residualt(veoxk, veoyk, Tek, residualtek, kx, ky);

				// pot source
				calcPotSourcek_inertia(ne0k, nek, dndx0k, dndy0k, dndxk, dndyk, dphidx0k, dphidy0k, dphidx1k, dphidy1k, Pi1k, Pe1k, uxB, e, Cm, hallEk, hallIk, vexbkx0, vexbky0, vexbkx, vexbky, kx, ky, ksqu, potSourcek_inertia);
				// c2rfft(potSourcek_inertia, Testt);
				// print2DB(neinitialg, Testt);
				// double ne0k[][ncomp], double nek[][ncomp],double dndx0k[][ncomp], double dndy0k[][ncomp], double dndxk [][ncomp], double dndyk [][ncomp], double dphidx0k [][ncomp], double dphidy0k [][ncomp], double dphidx1k [][ncomp], double dphidy1k [][ncomp], double Pi1k[][ncomp], double Pe1k[][ncomp], double uxB[], double e, double Cm, double hallEk [][ncomp], double hallIk [][ncomp], double vexbkx0[][ncomp], double vexbky0[][ncomp], double vexbkx[][ncomp],  double vexbky[][ncomp], double kx[], double ky[], double ksqu[], double potSourcek_inertia[][ncomp]){

				// plot
				phi_iter = potentialk(invnk, dndxk, dndyk, dphikdt, potSourcek_inertia, kx, ky, ninvksqu, err_max, phi_iter_max); // no difference b/w potentialk and the inertial one

				if (phi_iter > phi_iter_max)
				{
					printf("Blew up");
					break;
				}

				calc_sourcen(ksqu, ne1k, Dart, sourcen1k); //
				// RK ste[p

				RK4(ne1k_old, dt, residualnk, sourcen1k, stage, ne1k); //

				// RK4(phi1k_old, dt, residualk_phi, potSourcek_inertia, stage, phi1k); // potent now has time deriv so we need to RK it
				// try
				RK4(phi1k_old, dt, residualk_phi, dphikdt, stage, phi1k);

				RK4(Tik_old, dt, residualtik, sourcetk, stage, Tik);
				// ADD this test
				RK4(Tek_old, dt, residualtek, sourcetk, stage, Tek);
				// tot ne
				Arr3DArr3DAdd(ne0k, ne1k, nek);
				// phi
				Arr3DArr3DAdd(phi0k, phi1k, phik);

				// Calculate pressures
				convolve2D(nek, Tek, Pek);
				rscalArr3DMult(Pek, kb, Pek);

				convolve2D(nek, Tik, Pik);
				rscalArr3DMult(Pik, kb, Pik);
				// pert pressure
				Arr3DArr3DSub(Pek, Pe0k, Pe1k);
				Arr3DArr3DSub(Pik, Pi0k, Pi1k);
				// all derv for tot
				derivk(nek, kx, dndxk);
				derivk(nek, ky, dndyk);

				derivk(phik, kx, dphidxk);
				derivk(phik, ky, dphidyk);
				derivk(Pek, kx, dpedxk);
				derivk(Pek, ky, dpedyk);
				derivk(Pik, kx, dpidxk);
				derivk(Pik, ky, dpidyk); //
				// deriv pert
				// add phi1k
				derivk(phi1k, kx, dphidx1k);
				derivk(phi1k, ky, dphidy1k);
				// collision freq
				calcCollFreqk_inertia(nek, Tik, Tek, kb, eps0, mi, me, ri, rn, nn, Oci, Oce, e, nuink, nuiek, nuiik, nuenk, nueek, nueik, isigPk, invnk, hallIk, hallEk);

				// tot velo
				calcV_ExBk(dphidxk, dphidyk, B, B2, vexbkx, vexbky);
				// diam for e anf i tot
				calc_diamag(dpedxk, dpedyk, B, B2, -1 * e, nek, vdmexk, vdmeyk);
				calc_diamag(dpidxk, dpidyk, B, B2, e, nek, vdmixk, vdmiyk);

				// Get total velocity

				Arr3DArr3DAdd(vdmexk, vexbkx, veoxk);
				Arr3DArr3DAdd(vdmeyk, vexbky, veoyk);
				Arr3DArr3DAdd(vdmixk, vexbkx, vioxk);
				Arr3DArr3DAdd(vdmiyk, vexbky, vioyk);
			}
			//% Save mesh
			// save('X.txt','XX','-ascii');
			// save('Y.txt','YY','-ascii');
			// Check for convergence with potential. We can go over this part together later.: we need 2 checks for convergence
			if (phi_iter > phi_iter_max)
			{
				printf("Blew up");
				break;
			}
			ofstream myfile;
			myfile.open("tVec.txt");
			for (int iter = -1; iter < iter_max; iter++)
			{
				myfile << std::setprecision(16) << time[iter + 1] << '\n';
			}
			myfile.close();

			// output time stamp
			printf("Iteration = %d    t = %.10f   phi_iter = %d\n", iter, time[iter + 1], phi_iter);
			// printf("%0.1f\n",(iter/saveFrequency) - saveNum);
			//  Save data every saveFrequency time steps
			if ((iter / saveFrequency) - saveNum == 0)
			{

				c2rfft(nek, ne);
				c2rfft(Tek, Te);
				c2rfft(Tik, Ti);
				c2rfft(phik, phi);
				// TEST PERTURBED
				// c2rfft(ne1k, ne1);
				// c2rfft(Tek, Te);
				// c2rfft(Tik, Ti);
				// c2rfft(phi1k, phi1);

				// char KEfilename[16] = {0};
				// char save[3];
				char save[16] = {0};
				snprintf(save, 16, "%d", saveNum);
				const char *type = ".txt"; // test from .txt
				const char *typeg = ".gkyl";

				char nefilename[16] = {0};
				strcat(nefilename, "ne");
				strcat(nefilename, save);
				strcat(nefilename, type);

				// ##################### TEST print both
				char nefilenameg[16] = {0};
				strcat(nefilenameg, "ne");
				strcat(nefilenameg, save);
				strcat(nefilenameg, typeg);

				// c2rfft(vexbky0, test); // FREE TEST
				// print2DB(testfilename, test);

				// char Tefilename[5];
				char Tefilename[16] = {0};
				strcat(Tefilename, "Te");
				strcat(Tefilename, save);
				strcat(Tefilename, type);
				// snprintf(Tefilename, 16, "%d", saveNum);
				// strcat(Tefilename, type); //this was commented

				// ##################### TEST print both
				char Tefilenameg[16] = {0};
				strcat(Tefilenameg, "Te");
				strcat(Tefilenameg, save);
				strcat(Tefilenameg, typeg);

				// char Tifilename[5];
				char Tifilename[16] = {0}; // s an array statically allocated (so it's size is defined and fixed during compilation
				strcat(Tifilename, "Ti");
				strcat(Tifilename, save);
				strcat(Tifilename, type);
				// snprintf(Tifilename, 16, "%d", saveNum);
				// strcat(Tifilename, type); //this was commented

				// ##################### TEST print both
				char Tifilenameg[16] = {0}; // s an array statically allocated (so it's size is defined and fixed during compilation
				strcat(Tifilenameg, "Ti");
				strcat(Tifilenameg, save);
				strcat(Tifilenameg, typeg);

				// char phifilename[5];
				char phifilename[16] = {0};
				strcat(phifilename, "phi");
				strcat(phifilename, save);
				strcat(phifilename, type);
				// snprintf(phifilename, 16, "%d", saveNum);
				// strcat(phifilename, type); //this was commented
				// print2DArrf(Nfilename, std::string("N") + std::to_string(saveNumber) + ".txt");

				// ##################### TEST print both
				char phifilenameg[16] = {0};
				strcat(phifilenameg, "phi");
				strcat(phifilenameg, save);
				strcat(phifilenameg, typeg);

				// print2DArrf(nefilename, std::string("ne") + std::to_string(saveNum) + ".txt");
				// print2DArrf(Tefilename, std::string("Te") + std::to_string(saveNum) + ".txt");
				// print2DArrf(Tifilename, std::string("Ti") + std::to_string(saveNum) + ".txt");
				// print2DArrf(phifilename,std::string("phi") + std::to_string(saveNum) + ".txt");
				//  post gkyl here

				// print2DArrf(Tefilename, Te);
				// print2DArrf(Tifilename, Ti);
				// print2DArrf(phifilename, phi);
				// print2DArrf(nefilename, ne);

				// print1DArrf(timefilename, time,iter_max);
				print2DArrf(Tefilename, Te);
				print2DArrf(Tifilename, Ti);
				print2DArrf(phifilename, phi);
				print2DArrf(nefilename, ne);

				// test
				print2DB(Tefilenameg, Te);	 // test
				print2DB(Tifilenameg, Ti);	 // test
				print2DB(phifilenameg, phi); // test
				print2DB(nefilenameg, ne);	 // test

				memset(nefilename, 0, sizeof nefilename);
				memset(Tefilename, 0, sizeof Tefilename);
				memset(Tifilename, 0, sizeof Tifilename);
				memset(phifilename, 0, sizeof phifilename);
				// added

				saveNum++;
			}
			// If end time is reached, end simulation

			// If end time is reached, end simulation
		}
	*/

	fftw_free(XX);
	fftw_free(YY);
	free(kx);
	free(ky);
	free(ksqu);
	free(ninvksqu);
	free(ne);
	free(nek);
	free(Ti);
	free(Tik);
	free(Te);
	free(Tek);
	free(phi);
	free(phik);
	free(Pi);
	free(Pik);
	free(Pe);
	free(Pek);
	free(vexbx);
	free(vexbkx); // change name here too
	free(vexby);
	free(vexbky); // change name here too
	free(vdmex);
	free(vdmexk);
	free(vdmix);
	free(vdmixk);
	free(vdmiy);
	free(vdmiyk);
	free(vdmey);
	free(vdmeyk);
	free(veoxk);
	free(vioxk);
	free(vioyk);
	free(veoyk);
	free(residualnk);
	// free(test);
	free(residualtik);
	free(residualtek);
	// free(sourcenk); //not used for now
	free(sourcetk);
	free(nek_old);
	free(Tik_old);
	free(Tek_old);
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
	free(hallEk); // TEST for inertial
	free(hallIk);
	free(dndx0k);
	free(dndx1k);
	free(dndy0k);
	free(dndy1k);
	free(ne0k);
	free(ne1k);
	free(vexbkx0);
	free(vexbkx1);
	free(vexbky0);
	free(vexbky1);
	free(d2phidxk);
	free(d2phidyk);
	free(dphidx0k);
	free(dphidx1k);
	free(dphidy0k);
	free(dphidy1k);
	free(Pe1k); // did not use rtcfft so might not work
	free(Pi1k); // not sure
	// free(uxB_x);
	// free(uxB_y);
	// free(uxB);
	free(phi1k);
	free(phi0k);
	free(potSourcek_inertia);
	free(Pe0);
	free(Pi0);
	free(Pe1);
	free(Pi1);
	free(Ti_0);
	free(Te_0);
	free(ne1);
	free(ne0);
	free(phi1);
	free(phi_0);
	free(Pi0k);
	free(Pe0k);
	free(Ti0k); // test
	free(Te0k);
	// free(potSource_inertia); //test
	free(vexbx0);
	free(vexbx1);
	free(vexby1);
	free(vexby0);
	// new residuals
	// free(residualn1k);
	free(residualk_phi);
	free(vdmex1k);
	free(vdmey1k);
	free(vdmix1k);
	free(vdmiy1k);
	free(vdmiy1);
	free(vdmix1);
	free(vdmey1);
	free(vdmex1);
	free(veox1k);
	// free(viox1k);
	// free(veoy1k); // not used for now
	// free(vioy1k);
	free(ne1k_old);
	free(phi1k_old);
	free(dpedx1k);
	free(dpedy1k);
	free(dpidx1k);
	free(dpidy1k);
	free(dphikdt);
	free(Ti_1);
	free(Te_1);
	free(sourcen1k);

	return 0;
}
