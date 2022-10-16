#define _USE_MATH_DEFINES // for C++
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <string>
#include "spectralFunctions.h"
#include "fftw3.h"
#include "arithmeticFunctions.h"
#include <cstring>
#include<stdint.h>
#include <sstream>
#include <iostream> //test
#include <fstream>
#include <vector>
#include <iterator>
//#include <fstream> //for file operations
#include <iomanip>

using namespace std;
// This cpp file make some simple arithmetic functions that will be useful in the code.
// Feel free to add to this as new operations come up and you feel they belong here.

// Prints a 1D array in a column
void print1DArray(double arr[], int nArr){
	// Get size of the 1D array first.
	for (int i = 0; i < nArr; i++) {
		printf("%+4.2le\n",arr[i]);
	}

}



// Prints a 2D array as a matrix (can probably be made to look prettier. A nicer one probably exists online).
// use this for the inv FF of potsours
// update indexing
void print2DPhysArray(double arr[]) { //from arr[][ny]
	for (int i = 0; i < nx; i++) {
		//std::cout << i << "\t";
		for (int j = 0 ; j < ny ; j++) {
			printf("%+4.2le  ",arr[j + ny*i]);//"\n"); //[j + ny*i] real. from arr[i][j]). from: printf("%+4.2le  ",arr[j + ny*i]);
			//printf("\n"); //one LONG coloumn wrong
		}
		printf("\n");
		//std::cout << "\n";
	}
}

// Prints a 2D array as a matrix (can probably be made to look prettier. A nicer one probably exists online).
void print2DFourierArray(double arr[][nyk]) {
	for (int i = 0; i < nx; i++) {
		for (int j = 0 ; j < nyk ; j++) {
			for(int k = 0; k < ncomp; k++){
				printf("%+4.2le  ",arr[j + nyk*i][k]); //[j + nyk*i][k] fourier. from [i][j]
		    }
				//from here: printf("\n");
	    }
		printf("\n");
    }
}

// Prints a 3D array as a set of 2 matrices (can probably be made to look prettier. A nicer one probably exists online).
void print3DFourierArray(double arr[][nyk][ncomp]) {
	printf("Real:\n");
	for (int i = 0; i < nx; i++) {
		for (int j = 0 ; j < nyk ; j++) {
			printf("%+4.2le  ",arr[i][j][0]); // print3DFourierArray(potSourcek_inertia)
		}
		printf("\n");
	}

	printf("\nImag:\n");
	for (int i = 0; i < nx; i++) {
		for (int j = 0 ; j < nyk ; j++) {
			printf("%+4.2le  ",arr[i][j][1]);
		}
		printf("\n");
	}
}

// Create a hyperbolic secant function used for testing purposes
double sech(double val){
	return 1/cosh(val);
}

// This function calculates the magnitude of a 1D array with 3 elements
//added this not sure since contains 3 elements
double mag1DArray(double arr[]){
	return sqrt(arr[0]*arr[0] + arr[1]*arr[1] + arr[2]*arr[2]);
}

// This function multiplies a 2D physical array by a scalar. Division handled by just modifying scal outside of the function.
//double arrOut[16] = {0}; //added this
void scalArr2DMult(double arr[], double scal, double arrOut[]){
	//#pragma omp parallel for schedule(static) collapse(2)
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < ny; j++){
			arrOut[j + ny*i] = arr[j + ny*i] * scal;
		}
	}
}

// This function multiplies a 2D Fourier array by a scalar
void scalArr2DMultk(double arr[][nyk], double scal, double arrOut[][nyk]){
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < nyk; j++){
			arrOut[i][j] = arr[i][j] * scal;
		}
	}
}

// This function adds a scalar value to a 2D physical array
//added this not sure about size
void scalArr2DAdd(double arr[], double scal, double arrOut[]){
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < ny; j++){
			arrOut[j + ny*i] = arr[j + ny*i] + scal;
		}
	}
}

// This function adds a scalar value to a 2D Fourier array
void scalArr2DAddk(double arr[][nyk], double scal, double arrOut[][nyk]){
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < nyk; j++){
			arrOut[i][j] = arr[i][j] + scal;
		}
	}
}

// This function multiplies a 3D Fourier array by a real valued scalar. Handle division by modifying rscal outside of function.
void rscalArr3DMult(double arr[][ncomp], double rscal, double arrOut[][ncomp]){
	#pragma omp parallel for schedule(static) collapse(3)
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < nyk; j++){
			for (int k = 0; k < ncomp; k++){
				arrOut[j + nyk*i][k] = arr[j + nyk*i][k] * rscal;
			}
		}
	}
}

// This function multiplies a 3D Fourier array by i.
void iArr3DMult(double arr[][ncomp], double arrOut[][ncomp]){
	//double dummy;   // This is used so that data are not accidentally overwritten
	#pragma omp parallel for schedule(static) collapse(2)
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < nyk; j++){
			const double dummy = arr[j + nyk*i][0];
			arrOut[j + nyk*i][0] = -arr[j + nyk*i][1];
			arrOut[j + nyk*i][1] =  dummy;
		}
	}


}

// This function adds a real valued scalar to a 3D Fourier array.
void rscalArr3DAdd(double arr[][ncomp], double rscal, double arrOut[][ncomp]){
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < nyk; j++){
				arrOut[j + nyk*i][0] = arr[j + nyk*i][0] + rscal;
				arrOut[j + nyk*i][1] = arr[j + nyk*i][1];
		}
	}
}

// This function adds an imaginary valued scalar to a 3D Fourier array.
void iscalArr3DAdd(double arr[][nyk][ncomp], double iscal, double arrOut[][nyk][ncomp]){
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < nyk; j++){
				arrOut[i][j][0] = arr[i][j][0];
				arrOut[i][j][1] = arr[i][j][1] + iscal;
		}
	}
}

// This function multiplies, elementwise, a 2D physical array by a 2D physical array
void Arr2DArr2DMult(double arrIn0[], double arrIn1[], double arrOut[]){
	#pragma omp parallel for schedule(static) collapse(2)
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < ny; j++){
			arrOut[j + ny*i] = arrIn0[j + ny*i] * arrIn1[j + ny*i];
		}
	}
}
void Arr2DArr2DDiv(double arrIn0[], double arrIn1[], double arrOut[]){
	#pragma omp parallel for schedule(static) collapse(2)
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < ny; j++){
			arrOut[j + i*ny] = arrIn0[j + i*ny]/arrIn1[j + i*ny];
		}
	}
}

// This function adds, elementwise, a 2D physical array by a 2D physical array
void Arr2DArr2DAdd(double arrIn0[], double arrIn1[], double arrOut[]){
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < ny; j++){
			arrOut[j + ny*i] = arrIn0[j + ny*i] + arrIn1[j + ny*i];
		}
	}
}

// This function subtracts, elementwise, a 2D physical array by a 2D physical array.
void Arr2DArr2DSub(double arrIn0[], double arrIn1[], double arrOut[]){
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < ny; j++){
			arrOut[j + ny*i] = arrIn0[j + ny*i] - arrIn1[j + ny*i];
		}
	}
}

// This function multiplies a 3D Fourier array by a 2D real valued array.
void Arr3DArr2DMult(double arr3D[][ncomp], double arr2D[], double arrOut[][ncomp]){
	#pragma omp parallel for schedule(static) collapse(3)
	for(int i = 0; i < nx; i++){
		for(int j = 0; j < nyk; j++){
			for (int k = 0; k < ncomp; k++){
				arrOut[j + nyk*i][k] = arr3D[j + nyk*i][k] * arr2D[j + nyk*i];
			}
		}
	}
}

// This function divides a 3D Fourier array by a 2D real valued array.
void Arr3DArr2DDiv(double arr3D[][ncomp], double arr2D[], double arrOut[][ncomp]){
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < nyk; j++){
			for (int k = 0; k < ncomp; k++){
				arrOut[j + nyk*i][k] = arr3D[j + nyk*i][k] / arr2D[j + nyk*i];
			}
		}
	}
}

// This function adds 2 3D Fourier arrays.
void Arr3DArr3DAdd(double arr0[][ncomp], double arr1[][ncomp], double arrOut[][ncomp]){
	#pragma omp parallel for schedule(static) collapse(3)
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < nyk; j++){
			for( int k = 0; k < ncomp; k++){
				arrOut[j + nyk*i][k] = arr0[j + nyk*i][k] + arr1[j + nyk*i][k];
			}
		}
	}
}
void Arr3DArr3DMult(double arr0[][ncomp], double arr1[][ncomp], double arrOut[][ncomp]){
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < nyk; j++){
			for( int k = 0; k < ncomp; k++){
				arrOut[j + nyk*i][k] = arr0[j + nyk*i][k] * arr1[j + nyk*i][k];
			}
		}
	}
}

// This function subtracts a 3D Fourier arrays from another 3D Fourier array.
// It will always be arrIn0 - arrIn1
void Arr3DArr3DSub(double arrIn0[][ncomp], double arrIn1[][ncomp], double arrOut[][ncomp]){
	#pragma omp parallel for schedule(static) collapse(3)
	for(int i = 0; i < nx; i++){
		for( int j = 0; j < nyk; j++){
			for( int k = 0; k < ncomp; k++){
				arrOut[j + nyk*i][k] = arrIn0[j + nyk*i][k] - arrIn1[j + nyk*i][k];
			}
		}
	}
}

// This function calculates the absolute value of a complex number. This is equivalent to the modulus.
void absComp(double arr3D[][ncomp], double arrOut[]){
	for (int i = 0; i < nx ; i++){
		for (int j = 0 ; j < nyk; j++){
			arrOut[j + nyk*i] = sqrt(arr3D[j + nyk*i][0]*arr3D[j + nyk*i][0] + arr3D[j + nyk*i][1]*arr3D[j + nyk*i][1]);
		}
	}
}

// This functions calculates the maximum value of a 2D Fourier array
//nx nyk size of complex 512x257 and real is nxny
double max2Dk(double arr2D[]){
	double maxVal = 0.;
	for (int i = 0; i < nx; i++){
		for (int j = 0; j < nyk; j++){
			if (arr2D[j + nyk*i] > maxVal){ //if (n)--> n isn't initialized here at
				//double *arr2D= new double [j+nyk*i]; //test this isn't initialized

				maxVal = arr2D[j + nyk*i];
			}
		}
	}
	return maxVal;
}
//For multidimensional arrays in C/C++ you have to specify all dimensions except the first
double max2D(double arr2D[]){
	double maxVal = 0.;
	for (int i = 0; i < nx; i++){
		for (int j = 0; j < ny; j++){
			if (arr2D[j + ny*i] > maxVal){
				maxVal = arr2D[j + ny*i];
			}
		}
	}
	return maxVal;
}

// This function combines the previous two functions. Takes the absolute value and then gives the max
//double arr3D[16]={0}; //added this to fix initializing
double max_absComp(double arr3D[][ncomp]){
	// Take the absolute value
	double *absArr;
	absArr = (double*) fftw_malloc(nx*nyk*sizeof(double));
	memset(absArr, 42, nx*nyk* sizeof(double)); //test here, if you pass 2D array to func decays to pointer and sizeof doesn't give size of array


	absComp(arr3D, absArr);

	// Calculate the max value
	double maxVal = max2Dk(absArr); //by
	fftw_free(absArr);
	//free(absArr);

	return maxVal;

}
//double arr[16] = {0}; //added this to try initialize
double Arr1DMax(double arr[], int arrLength){
	double max = 0.0;
	for (int i = 0; i < arrLength; i++){
		if (arr[i] > max){
			max = arr[i];
		}
	}
	return max;
}
//double approx [16] = {0}; //this was added to try initialize
//double exact [16] = {0}; //this was added
double l2norm(double approx [][ny], double exact [][ny], double p){
	double sum = 0;
	double sum2 = 0;
	for (int i = 0; i < nx; i++){
		for (int j = 0; j < ny; j++){
			double error = (exact[i][j] - approx[i][j]);
			error = pow(error, p);
			sum += error;
		if (exact[i][j] >0){
			sum2 += exact[i][j];
		} else{
			sum2 -= exact[i][j];
		}
		}
	}

	sum2 /= nx*ny;

	return (pow(sum/(nx*ny), 1/p))/sum2;
}
//TEST: write 1D vector into file

//void print1DArrf(char filename[], double arr[], int nArr){
void print1DVec(const std::string& filename, const std::vector<double>& vec){

	std::ofstream myfile;
	myfile.open (filename);
  	myfile << std::setprecision(16) <<  &vec << '\n';
	//printf("t = %.10f  \n", &vec);
  	myfile.close();

	//TEST
	/*
	ofstream myfile;
	myfile.open (filename);
  			//myfile.open ("tVec.txt");
			for (iter = -1; iter < itermax; iter++){
				myfile << std::setprecision(16) << &vec[iter+1] << '\n';
			}
  			myfile.close();
	*/
	//fptr = fopen(filename, "w");
	//std::string name = "tVec.txt";
    //std::ofstream out(name);
	//output_iterator(out, "\n");
	//std::ofstream out(name.c_str());
	/*
	std::ofstream ofs(filename, std::ios::out | std::ofstream::binary);
	std::ostream_iterator<char> osi{ofs};
	const char* beginByte = (char*)&vec[0];
	const char* endByte = (char*)&vec.back() + sizeof(double);
	std::copy(beginByte, endByte, osi);
	*/

	//ofstream myfile;
 	//myfile.open ("tVec.txt");
  	//myfile << "t = %.16f \n",time[iter+1];
	 //printf("Iteration = %d    t = %.10f   phi_iter = %d\n", iter, time[iter+1], phi_iter);
  	//myfile.close();


	// Prints a 1D array in a column
 //void print1DArray(double arr[], int nArr){

	// Get size of the 1D array first.

	//for (int i = 0; i < nArr; i++) {
	//	fprintf(fptr,"%+4.16le\n",arr[i]);
	//}

	//fclose(fptr);
}
// try
//output time stamp	printf("Iteration = %d    t = %.10f   phi_iter = %d\n", iter, time[iter+1], phi_iter);


//write a 2D array to file
void print2DArrf(char filename[], double arr[]){

	FILE *fptr;

	fptr = fopen(filename, "w");


	for (int i = 0; i < nx; i++) {
		for (int j = 0 ; j < ny ; j++) {
			fprintf(fptr, "%+4.16le  ",arr[j + ny*i]);
		}
		fprintf(fptr, "\n");
	}

	fclose(fptr);
}

//write 2D array in fourier space to file

void print2DFouArrf(char filename[], double arr[]){

	FILE *fptr;

	fptr = fopen(filename, "w");


	for (int i = 0; i < nx; i++) {
		for (int j = 0 ; j < nyk ; j++) {
			fprintf(fptr, "%+4.16le  ",arr[j + nyk*i]);
		}
		fprintf(fptr, "\n");
	}

	fclose(fptr);
}

// maybe try creating a cross product function: there are 2 ways to implement this in C, a function that returns a scalar or a vector. I will choose a vector to match the cross func in MATLAB
//function to calculate cross product of two vectors

void cross_product(double vector_a[], double vector_b[], double temp[]) {

   temp[0] = vector_a[1] * vector_b[2] - vector_a[2] * vector_b[1];
   temp[1] = vector_a[2] * vector_b[0] - vector_a[0] * vector_b[2];//   temp[1] = vector_a[0] * vector_b[2] - vector_a[2] * vector_b[0];
   temp[2] = vector_a[1] * vector_b[0] - vector_a[0] * vector_b[1];// temp[2] = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0];

}

//
void print2DB(char filename[], double arr[]){ //binary

    // open file to write solution
    FILE *fp = fopen(filename, "wb"); if (!fp) return;

    // write grid
    uint64_t ndim = 2;
    uint64_t cells[] = { nx,ny };
    double lower[] = { 0,0 }, upper[] = { 1,1 }; //  test

	uint64_t real_type = 2;
    fwrite(&real_type, sizeof(uint64_t), 1, fp);

    fwrite(&ndim, sizeof(uint64_t), 1, fp);
    fwrite(cells, 2*sizeof(uint64_t), 1, fp);
    fwrite(lower, 2*sizeof(double), 1, fp);
    fwrite(upper, 2*sizeof(double), 1, fp);

    uint64_t esznc = sizeof(double), size =nx*ny;
    fwrite(&esznc, sizeof(uint64_t), 1, fp);
    fwrite(&size, sizeof(uint64_t), 1, fp);

    fwrite(&arr[0], esznc*size, 1, fp);
    fclose(fp);
}

//double absolute(double arr[], int arrLength){
	//int i;
//	double min = 0.0;
//	for(int i=0; i< arrLength; i++){
//		if (arr[i] < min){
//			min = arr[i] * (-1);
//		}
//	}
//	return min;
//}

void absolute(double arr[], double arrOut[]){
	int N=3;
	for (int i = 0; i < N ; i++){
		//for (int j = 0 ; j < nyk; j++){
			if (arr[i] < 0){
				arrOut[i] = arr[i] * (-1);

			}


	}
}
