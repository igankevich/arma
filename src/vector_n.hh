#ifndef VECTOR_N_HH
#define VECTOR_N_HH

#include <ostream>
#include <istream>

namespace autoreg {

/// @n-component vector.
template <class T, size_t n>
class Vector {
	T coord[n];

public:
	typedef T Value;

	Vector() {
		for (size_t i=0; i<n; ++i)
			coord[i] = T(0);
	}

	template <class A>
	explicit
	Vector(const Vector<A, n> &v) {
		for (size_t i=0; i<n; ++i)
			coord[i] = v[i];
	}

	template <class A>
	Vector(A x, A y, A z) {
		coord[0] = x; coord[1] = y; coord[2] = z;
	}

	template <class A>
	Vector(A x, A y) {
		coord[0] = x; coord[1] = y;
	}

	template <class A>
	explicit Vector(A x) { 
		for (size_t i=0; i<n; ++i) {
			coord[i] = x; 
		}
	}

	Vector operator+(const Vector& rhs) const { return trans(rhs, std::plus<T>()); }
	Vector operator-(const Vector& rhs) const { return trans(rhs, std::minus<T>()); }
	Vector operator*(const Vector& rhs) const { return trans(rhs, std::multiplies<T>()); }
	Vector operator/(const Vector& rhs) const { return trans(rhs, std::divides<T>()); }
	
	Vector operator+(T val) { return trans(std::bind2nd(std::plus<T>(), val)); }
	Vector operator-(T val) { return trans(std::bind2nd(std::minus<T>(), val)); }
	Vector operator*(T val) { return trans(std::bind2nd(std::multiplies<T>(), val)); }
	Vector operator/(T val) { return trans(std::bind2nd(std::divides<T>(), val)); }

	Vector& operator=(const T& rhs) {
	    for (size_t i=0; i<n; ++i)
            coord[i] = rhs;
	    return *this;
	}

	Vector& operator=(const Vector& rhs) {
		for (size_t i=0; i<n; ++i)
			coord[i] = rhs.coord[i];
		return *this;
	}

	bool operator==(const Vector<T, n> &v) const {
		for (size_t i=0; i<n; ++i)
			if (coord[i] != v.coord[i])
				return false;
		return true;
	}

	bool operator!=(const Vector& rhs) const { return !operator==(rhs); }

	bool operator<(const Vector<T, n> &v) const {
		for (size_t i=0; i<n; ++i) {
			if (coord[i] < v.coord[i]) return true;
			if (coord[i] > v.coord[i]) return false;
		}
		return false;
	}

	void rot(size_t r) {
		std::rotate(&coord[0], &coord[r%n], &coord[n]);
	}

	T& operator[] (size_t i) { return coord[i]; }
	const T& operator[] (size_t i) const { return coord[i]; }

//	operator size_t() const { return size(); }
	T count() const {
		return reduce(std::multiplies<T>());
	}

	template<class BinOp>
	T reduce(BinOp op) const {
    	T s = coord[0];
    	for (size_t i=1; i<n; ++i)
    	    s = op(s, coord[i]);
    	return s;
	}

	friend std::ostream&
	operator<<(std::ostream& out, const Vector& rhs) {
		out << rhs.coord[0];
		for (size_t i=1; i<n; ++i) {
			out << ',' << rhs.coord[i];
		}
		return out;
	}

	friend std::istream&
	operator>>(std::istream& in, Vector& rhs) {
		for (size_t i=0; i<n; ++i) {
			in >> rhs.coord[i];
			in.get();
		}
		return in;
	}

private:

	template <class BinOp>
	Vector trans(const Vector& v, BinOp op) const {
		Vector v2;
		std::transform(coord, coord+n, v.coord, v2.coord, op);
		return v2;
	}

	template <class UnOp>
	Vector trans(UnOp op) const {
		Vector new_vec;
		std::transform(coord, coord+n, new_vec.coord, op);
		return new_vec;
	}

};

template<typename T>
class Vector<T, 1> {
	T val;
public:
	Vector(): val(0) {}
	template <class A> Vector(A t): val(t) {}
	operator T&() { return val; }
	operator const T&() const { return val; }
	T& operator[] (size_t) { return val; }
	const T& operator[] (size_t) const { return val; }
};

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
