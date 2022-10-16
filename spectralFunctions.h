#ifndef spectralFUN_H
#define spectralFUN_H

// Define array sizes here for physical variables
static const int nx = 10000;
static const int ny = 1000;

// static const int nx = 256; //256; //1024; //2048; // 256;//512; //256;//4 x 4; //16; //try 16x16 for printing out purposes
// static const int ny = 256;//256; //512;//512; //16;

// Define array sizes here for Fourier variables
//static const int nxk = nx;     // Not sure if I even need this
static const int nyk = ny/2 + 1;
static const int ncomp = 2;

double makeSpatialMesh2D(double dx, double dy, double xarr[], double yarr[]);
//void makeFourierMesh2D(double Lx, double Ly, double KX[], double KY[], double ksqu[], double ninvksqu[]);
void makeFourierMesh2D(double Lx, double Ly, double KX[], double KY[], double ksqu[], double ninvksqu[], int FourierMeshType);
void r2cfft(double rArr[], double cArr[][ncomp]);
void c2rfft(double cArr[][ncomp], double rArr[]);
void derivk( double vark[][ncomp], double k[], double derivative[][ncomp]);
void laplaciank( double vark[][ncomp], double ksqu[], double derivative[][ncomp]);
void convolve2D( double fk[][ncomp], double gk[][ncomp], double fgk[][ncomp]);
//changed calcfreqk from: calcCollFreqk(nek, Tik, Tek , kb, eps0, mi, me, ri, rn, nn, Oci, Oce, e, nuink, nuiek, nuiik, nuenk, nueek, nueik, isigPk, invnk);
void calcCollFreqk_inertia( double nek[][ncomp], double Tik[][ncomp], double Tek[][ncomp] , double kb, double eps0, double mi, double me, double ri, double rn, double nn, double Oci, double Oce, double e, double nuink[][ncomp], double nuiek[][ncomp], double nuiik[][ncomp], double nuenk[][ncomp], double nueek[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double invnk[][ncomp], double hallIk[][ncomp], double hallEk[][ncomp]);
void calcCollFreqk( double nek[][ncomp], double Tik[][ncomp], double Tek[][ncomp] , double kb, double eps0, double mi, double me, double ri, double rn, double nn, double Oci, double Oce, double e, double nuink[][ncomp], double nuiek[][ncomp], double nuiik[][ncomp], double nuenk[][ncomp], double nueek[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double invnk[][ncomp]);	// Take inverse fft of nek, Tik, and Tek. 

//void calcCollFreqk( double nek[][ncomp], double Tik[][ncomp], double Tek[][ncomp] , double kb, double eps0, double mi, double me, double ri, double rn, double nn, double Oci, double Oce, double e, double nuink[][ncomp], double nuiek[][ncomp], double nuiik[][ncomp], double nuenk[][ncomp], double nueek[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double invnk[][ncomp]);
void calcPotSourcek(double dndxk[][ncomp], double dndyk[][ncomp], double Pik[][ncomp], double Pek[][ncomp], double nuink[][ncomp], double nuiek[][ncomp], double nuenk[][ncomp], double nueik[][ncomp], double isigPk[][ncomp], double Oci, double Oce, double u[], double B[], double ksqu[], double potSourcek[][ncomp]);
int potentialk(double invnk[][ncomp], double dndxk[][ncomp], double dndyk[][ncomp], double phik[][ncomp], double potSourcek[][ncomp], double kx[], double ky[], double ninvksqu[], double err_max, int max_iter);
// ********************** inertial TEST ***************
int potentialk_inertia(double invnk[][ncomp], double dndxk[][ncomp], double dndyk[][ncomp], double phik[][ncomp], double potSourcek_inertia[][ncomp], double kx[], double ky[], double ninvksqu[], double err_max, int max_iter);

void calcV_ExBk(double dphidxk[][ncomp], double dphidyk[][ncomp], double B[], double B2, double vexbkx[][ncomp], double vexbky[][ncomp]);
void fourierDivision2D(double fk[][ncomp], double gk[][ncomp], double fgk[][ncomp]);
void calc_residualt(double voxk[][ncomp], double voyk[][ncomp], double tempink[][ncomp], double tempoutk[][ncomp], double kx[], double ky[]);
void calc_residualn(double vexbxk[][ncomp], double vexbyk[][ncomp], double nink[][ncomp], double residnoutk[][ncomp], double kx[], double ky[]);
void calc_sourcen(double ksqu[], double nk[][ncomp], double d, double sourcenk[][ncomp]);
void RK4(double f[][ncomp], double dt, double residual[][ncomp], double source[][ncomp], int stage, double fout[][ncomp]);

//double calc_dt (double vexbx[ny], double vexby[], double diamagxi[], double diamagyi[], double diamagxe[], double diamagye[], double cfl, double kmax, double maxdt);
double calc_dt (double U[], double vexbx[], double vexby[], double diamagxi[], double diamagyi[], double diamagxe[], double diamagye[], double cfl, double kmax, double maxdt);
void calc_diamag(double dpdxk[][ncomp], double dpdyk[][ncomp], double B[], double B2, double qa, double nak[][ncomp], double diamagxk[][ncomp], double diamagyk[][ncomp]);
//*****************TEST inertial****************
void calcPotSourcek_inertia(double ne0k[][ncomp], double nek[][ncomp],double dndx0k[][ncomp], double dndy0k[][ncomp], double dndxk [][ncomp], double dndyk [][ncomp], double dphidx0k [][ncomp], double dphidy0k [][ncomp], double dphidx1k [][ncomp], double dphidy1k [][ncomp], double Pi1k[][ncomp], double Pe1k[][ncomp], double uxB[], double e, double Cm, double hallEk [][ncomp], double hallIk [][ncomp], double vexbkx0[][ncomp], double vexbky0[][ncomp], double vexbkx[][ncomp],  double vexbky[][ncomp], double kx[], double ky[], double ksqu[], double potSourcek_inertia[][ncomp]);
//test for dphit
int ddt_potentialk(double invnk[][ncomp],double ninvksqu[], double dndxk[][ncomp], double dndyk[][ncomp], double dphikdt_old[][ncomp],double potSourcek[][ncomp], double kx[], double ky[], double err_max,int max_iter, double dphikdt[][ncomp] );
// max_iter is phi_iter_max
//int ddt_potentialk(double invnk[][ncomp],double ninvksqu[], double dndxk[][ncomp], double dndyk[][ncomp], double dphikdt_old[][ncomp],double potSourcek[][ncomp], double kx[], double ky[], double err_max, int phi_iter_max, double dphikdt[][ncomp], int max_iter );
//int ddt_potentialk(double invnk[][ncomp],double ninvksqu[], double dndxk[][ncomp], double dndyk[][ncomp], double dphikdt_old[][ncomp],double potSourcek[][ncomp], double kx[], double ky[], double err_max,  int max_iter, int phi_iter_max, double dphikdt[][ncomp] );



//There may be some bug in this function, but it appears to work properly.  
//In the tests that we ran, we found that the real values of the ressult did not match with the matlab code.
//However, the imaginary numbers appeared to match between the two.
//In addition, there are some fourier arrays inside the function that have real terms that match with the matlab code
//But the imaginary values do not match.  We believe this to be a rounding error of some kind, as the difference between the order of magintudes
//Are around e15 or e16.





#endif
