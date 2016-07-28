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
	int n = 10;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	linalg::Vector<T> rhs(n);
	blitz::firstIndex i;
	blitz::secondIndex j;
	lhs = (i == j); // identity matrix
	rhs = T(1);
	linalg::cholesky(lhs, rhs);
	std::cout << __func__ << std::endl << rhs << std::endl;
	assert(std::all_of(rhs.begin(), rhs.end(), [](T x) { return x == T(1); }));
}

template <class T>
void
test_matrix_1() {
	int n = 3;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	linalg::Vector<T> rhs(n);
	lhs = 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0;
	rhs = T(1);
	linalg::cholesky(lhs, rhs);
	std::cout << __func__ << std::endl << rhs << std::endl;
	assert(approximately_equals(rhs(0), T(1), T(1e-3)));
	assert(approximately_equals(rhs(1), T(0), T(1e-3)));
	assert(approximately_equals(rhs(2), T(1), T(1e-3)));
}

template <class T>
void
test_matrix_2() {
	int n = 3;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	linalg::Vector<T> rhs(n);
	lhs = 1.00, 0.50, 0.33, 0.50, 1.00, 0.50, 0.33, 0.50, 1.00;
	rhs = T(1);
	linalg::cholesky(lhs, rhs);
	std::cout << __func__ << std::endl << rhs << std::endl;
	assert(approximately_equals(rhs(0), T(0.60241), T(1e-3)));
	assert(approximately_equals(rhs(1), T(0.39759), T(1e-3)));
	assert(approximately_equals(rhs(2), T(0.60241), T(1e-3)));
}

int
main() {
	typedef float T;
	test_indentity<T>();
	test_matrix_1<T>();
	test_matrix_2<T>();
	return 0;
}
