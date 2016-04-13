#ifndef VECTOR_N_HH
#define VECTOR_N_HH

#include <blitz/array.h>

namespace autoreg {

template<class T, size_t N>
using Vector = blitz::TinyVector<T, N>;

typedef Vector<size_t, 3> size3;
typedef Vector<size_t, 2> size2;
typedef Vector<size_t, 1> size1;


template<size_t n=3> class Index;
//template<size_t n=3> class Index_r;
//template<size_t n=3> class Index_zyx;

template<>
class Index<3> {
	const size3& s;
public:
	Index(const size3& sz): s(sz) {}
	size_t x(size_t i) const { return ((((i)/s[2])/s[1])%s[0]); }
	size_t y(size_t i) const { return (((i)/s[2])%s[1]); }
	size_t t(size_t i) const { return ((i)%s[2]); }
	size_t operator()(size_t i, size_t j=0, size_t k=0) const {
		return (i*s[1] + j)*s[2] + k;
	}
	size_t operator()(const size3& v) const { return v[0]*s[1]*s[2] + v[1]*s[2] + v[2]; }
};

template<>
class Index<1> {
	const size1& s;
public:
	Index(const size1& sz): s(sz) {}
	size_t operator()(size_t) const { return s[0]; }
};


template<>
class Index<2> {
	const size2& s;
public:
	Index(const size2& sz): s(sz) {}
	int operator()(int i, int j) const {
		return i*s[1] + j;
	}
};

}

#endif // VECTOR_N_HH
