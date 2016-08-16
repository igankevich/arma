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
	linalg::Vector<T> rhs(n);
	lhs = (i == j); // identity matrix
	rhs = T(1);
	rhs = linalg::operator*(lhs, rhs);
	std::cout << __func__ << std::endl << rhs << std::endl;
	assert(all(rhs == T(1)));
}

template <class T>
void
test_matrix_1() {
	int n = 2;
	linalg::Matrix<T> lhs(blitz::shape(n, n));
	linalg::Vector<T> rhs(n);
	lhs = 1.0, 2.0, 3.0, 4.0;
	rhs = 1.0, 2.0;
	rhs = linalg::operator*(lhs, rhs);
	std::cout << __func__ << std::endl << rhs << std::endl;
	assert(approximately_equals(rhs(0), T(5), T(1e-3)));
	assert(approximately_equals(rhs(1), T(11), T(1e-3)));
}

int
main() {
	typedef float T;
	test_indentity<T>();
	test_matrix_1<T>();
	return 0;
}
