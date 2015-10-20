#ifndef DOMAIN_HH
#define DOMAIN_HH

#include <istream>
#include <ostream>
#include "vector_n.hh"

namespace autoreg {

/// Domain is the region of computation
/// with @lower and @upper bounds and @length
/// which stores the number of points along
/// each dimension.
template <class T, size_t D=3u, class S=size_t>
class Domain {
public:	
	typedef Vector<T, D> Vec;
	typedef Vector<S, D> Size;

	Domain(Vec mi, Vec ma, Size s): lower(mi), upper(ma), length(s) {}
	Domain(Vec d, Size s): lower(), upper(d*Vec(s)), length(s) {}
	Domain(size_t nx, size_t ny, size_t nt): lower(0, 0, 0), upper(nx-1, ny-1, nt-1), length(nx, ny, nt) {} 
	Domain(Size s): lower(), upper(s), length(s) {} 
	Domain(T x0, T x1, size_t i, T y0, T y1, size_t j, T t0, T t1, size_t k): lower(x0, y0, t0), upper(x1, y1, t1), length(i, j, k) {}
	Domain(): lower(), upper(), length(S(1)) {}
	template <class A> Domain(const Domain<A, D>& d): lower(d.lower), upper(d.upper), length(d.length) {} 

	size_t dimensions() const { return D; }
	const Vec& min() const    { return lower; }
	const Vec& max() const    { return upper; }
	const Size& count() const { return length; }
	Vec& min()                { return lower; }
	Vec& max()                { return upper; }
	Size& count()             { return length; }
	Vec delta() const         { return (upper-lower)/Vec(length); }
	size_t size() const       { return length.size(); }

	Vec
	point(Size idx) const {
		return lower + delta() * Vec(idx);
	}

	Domain
	operator*(T factor) const {
		return Domain(lower, lower+(upper-lower)*factor, length*factor);
	}
	
	Domain
	operator+(size_t addon) const {
		return Domain(lower, lower+((upper-lower)/length)*(length+addon), length+addon);
	}
	
	const Domain&
	operator=(const Domain& d) {
		lower = d.lower;
		upper = d.upper;
		length = d.length;
		return *this;
	}

	friend std::ostream&
	operator<<(std::ostream& out, const Domain& rhs) {
		return out << rhs.lower << '\n' << rhs.upper << '\n' << rhs.length;
	}

	friend std::istream&
	operator>>(std::istream& in, Domain& rhs) {
		return in >> rhs.lower >> rhs.upper >> rhs.length;
	}

private:
	Vec lower, upper;
	Size length;
};

}

#endif // DOMAIN_HH
