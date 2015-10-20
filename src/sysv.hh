#ifndef SYSV_HH
#define SYSV_HH

#include <valarray>
#include <iostream>

/// @file
/// C/C++ interface to ``sysv'' LAPACK routine.

extern "C" void ssysv_(char*, int*, int*, float*, int*, int*, float*, int*, float*, int*, int*); 
extern "C" void dsysv_(char*, int*, int*, double*, int*, int*, double*, int*, double*, int*, int*); 

template<class T>
void sysv(char type, int m, int nrhs, T* a, int lda, T* b, int ldb, int* info);

template<>
void sysv<float>(char type, int m, int nrhs, float* a, int lda, float* b, int ldb, int* info)
{
	int lwork = m;
	std::valarray<float> work(lwork);
	std::valarray<int> ipiv(m);
	ssysv_(&type, &m, &nrhs, a, &lda, &ipiv[0], b, &ldb, &work[0], &lwork, info);
}

template<>
void sysv<double>(char type, int m, int nrhs, double* a, int lda, double* b, int ldb, int* info)
{
	int lwork = m;
	std::valarray<double> work(lwork);
	std::valarray<int> ipiv(m);
	dsysv_(&type, &m, &nrhs, a, &lda, &ipiv[0], b, &ldb, &work[0], &lwork, info);
}

#endif // SYSV_HH
