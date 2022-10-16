#ifndef arithmeticFUN_H
#define arithmeticFUN_H
#include <string>
#include <vector>



void print1DArray(double arr[], int nArr);
void print2DPhysArray(double arr[]); //from arr[][ny]
void print2DFourierArray(double arr[][nyk]);
void print3DFourierArray(double arr[][nyk][ncomp]);
double sech(double val);
double mag1DArray(double arr[]);
void scalArr2DMult(double arr[], double scal, double arrOut[]);
void scalArr2DMultk(double arr[][nyk], double scal, double arrOut[][nyk]);
void scalArr2DAdd(double arr[], double scal, double arrOut[]);
void rscalArr3DMult(double arr[][ncomp], double rscal, double arrOut[][ncomp]);
void rscalArr3DMult(double arr[][nyk][ncomp], double rscal, double arrOut[][nyk][ncomp]);
void rscalArr3DAdd(double arr[][nyk][ncomp], double rscal, double arrOut[][nyk][ncomp]);
void iscalArr3DAdd(double arr[][nyk][ncomp], double iscal, double arrOut[][nyk][ncomp]);
void Arr2DArr2DMult(double arrIn0[], double arrIn1[], double arrOut[]);
void Arr2DArr2DAdd(double arrIn0[], double arrIn1[], double arrOut[]);
void iArr3DMult(double arr[][ncomp], double arrOut[][ncomp]);
void Arr2DArr2DSub(double arrIn0[], double arrIn1[], double arrOut[]);
void Arr3DArr2DMult(double arr3D[][ncomp], double arr2D[], double arrOut[][ncomp]);
void Arr3DArr2DDiv(double arr3D[][ncomp], double arr2D[], double arrOut[][ncomp]);
void Arr3DArr3DAdd(double arr0[][ncomp], double arr1[][ncomp], double arrOut[][ncomp]);
void Arr3DArr3DSub(double arrIn0[][ncomp], double arrIn1[][ncomp], double arrOut[][ncomp]);
void absComp(double arr3D[][ncomp], double arrOut[]);
double max2Dk(double arr2D[]);
double max_absComp(double arr3D[][ncomp]);
void Arr2DArr2DDiv(double arrIn0[], double arrIn1[], double arrOut[]);
double Arr1DMax(double arr[], int arrLength);
double max2D(double arr2D[]);
void print2DArrf(char filename[], double arr[]); //change to take std::string
double l2norm(double approx [][ny], double exact [][ny], double p);
void Arr3DArr3DMult(double arr0[][ncomp], double arr1[][ncomp], double arrOut[][ncomp]);
void cross_product(double vector_a[], double vector_b[], double temp[]); //test
void print2DB(char filename[], double arr[]);
//double absolute(double arr[], int arrLength); // test
void absolute(double arr[], double arrOut[]); //test
void print2DFouArrf(char filename[], double arr[]); //test
void print1DArrf(char filename[], double arr[], int nArr); //test for time vector
void print1DVec(const std::string& filename, const std::vector<double>& vec); //another test
//void print1DVec(const std::string& filename, const std::vector<double>& vec, int iter, int itermax);
#endif
