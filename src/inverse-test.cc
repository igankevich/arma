#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <cmath>

#include "linalg.hh"

template <class T>
bool
approximately_equals(T lhs, T rhs, T eps) {
	return std::abs(lhs - rhs) < eps;
}

template <class T>
void
test_indentity() {
	using namespace blitz::tensor;
	using blitz::all;
	int n = 10;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	lhs = (i == j); // identity matrix
	linalg::inverse(lhs);
	std::cout << __func__ << std::endl << lhs << std::endl;
	assert(all(lhs(i,i) == T(1)));
}

template <class T>
void
test_matrix_1() {
	int n = 2;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	lhs = 1.0, 2.0, 3.0, 4.0;
	linalg::inverse(lhs);
	std::cout << __func__ << std::endl << lhs << std::endl;
	assert(approximately_equals(lhs(0,0), T(-2), T(1e-3)));
	assert(approximately_equals(lhs(0,1), T(1), T(1e-3)));
	assert(approximately_equals(lhs(1,0), T(1.5), T(1e-3)));
	assert(approximately_equals(lhs(1,1), T(-0.5), T(1e-3)));
}

int
main() {
	typedef float T;
	test_indentity<T>();
	test_matrix_1<T>();
	return 0;
}
