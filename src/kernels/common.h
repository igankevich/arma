#ifndef KERNELS_COMMON_H
#define KERNELS_COMMON_H
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NEW_VEC_IMPL(x,y) x##y
#define NEW_VEC(x,y) NEW_VEC_IMPL(x, y)

typedef ARMA_REAL_TYPE T;
typedef NEW_VEC(ARMA_REAL_TYPE,2) T2;
typedef NEW_VEC(ARMA_REAL_TYPE,4) T3; // use cl_double4 instead of cl_double3
typedef NEW_VEC(ARMA_REAL_TYPE,3) RealT3;
typedef NEW_VEC(ARMA_REAL_TYPE,4) T4;

typedef int4 Shape3D; // use int3 instead of int4 due to aligning issues

typedef union {
	T3 vec;
	T elem[sizeof(T3)];
} T3_union;

typedef union {
	int3 vec;
	int elem[sizeof(int3)];
} int3_union;


#endif // KERNELS_COMMON_H
