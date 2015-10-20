#ifndef EMATH_HH
#define EMATH_HH

#include <cmath>
#include <valarray>
#include <iostream>
#include <iomanip>

#include "domain.hh"

/// @file
/// Ancillary mathematical routines (interpolation,
/// Hermite polynomials, special functions, factorial etc.)

namespace autoreg {

const size_t MAX_POLY_ORDER = 60;

/// Computation of Hermite polynomial H_n(x).
/// Implemented via recursive relation:
/// H_2(x) = x*H_1(x) - 1*H_0(x)
/// H_3(x) = x*H_2(x) - 2*H_1(x)
/// H_4(x) = x*H_3(x) - 3*H_2(x).
/// @n -- polynomial order
template<class T>
T hermite(size_t n, T x)
{
	if (n == 0) return 1;
	if (n == 1) return x;
	T h0 = 1;
	T h1 = x;
	T hn = 0;
	for (size_t i=2; i<=n; i++) {
		hn = x*h1 - (i-1)*h0;
		h0 = h1;
		h1 = hn;
	}
	return hn;
}

/// Polynomial class for symbolic computations.
template<class T>
class Poly {
	std::valarray<T> a;

public:
	Poly(): a() {}
	Poly(size_t order): a(order+1) {}
	Poly(const std::valarray<T>& coefs): a(coefs) {}
	Poly(const Poly<T>& rhs): a(rhs.a) {}
	~Poly() {}

	Poly<T>& operator=(const Poly<T>& rhs) {
		ensure_size(rhs.a.size());
		a = rhs.a;
		return *this;
	}

	Poly<T> operator*(const Poly<T>& rhs) const {
		Poly<T> tmp(a.size() + rhs.a.size() - 1);
		for(size_t i=0; i<a.size(); ++i)
			for(size_t j=0; j<rhs.a.size(); ++j)
				tmp[i+j] += a[i]*rhs[j];
		return tmp;
	}

	size_t order() const {
		return a.size()-1;
	}
	T& operator[](size_t i) {
		return a[i];
	}
	T operator[](size_t i) const {
		return a[i];
	}

	friend std::ostream&
	operator<<(std::ostream& out, const Poly<T>& rhs) {
		for (int i=rhs.order()-1; i>1; --i)
			out << std::setw(16) << std::showpos << std::right << rhs[i] << "x^" << std::noshowpos << i;
		if (rhs.order() > 0)
			out << std::setw(16) << std::showpos << std::right << rhs[1] << "x";
		if (rhs.order() >= 0)
			out << std::setw(18) << std::showpos << std::right << rhs[0];
		return out;
	}

private:
	void ensure_size(size_t sz) {
		if (a.size() != sz) a.resize(sz);
	}
};

/// Constructs symbolic Hermite polynomial of order @n.
template<class T>
Poly<T> hermite_poly(size_t n) {
	if (n == 0) {
		Poly<T> p(0);
		p[0] = 1;  // 1
		return p;
	}
	if (n == 1) {
		Poly<T> p(1);
		p[0] = 0;  // x
		p[1] = 1;
		return p;
	}
	// в порядке убывания степеней
	T h0[MAX_POLY_ORDER] = {1.0f};
	T h1[MAX_POLY_ORDER] = {1.0f, 0.0f};
	Poly<T> hn(n+1);
	size_t h1_size = 2;
	size_t h0_size = 1;
	size_t m = std::min(MAX_POLY_ORDER, n+1);
	for (size_t i=2; i<m; i++) {
		size_t hn_size = h1_size+1;
		for (size_t j=0; j<h1_size; j++)
			hn[j] = h1[j];
		hn[h1_size] = 0.0f;
		for (size_t j=0; j<h0_size; j++)
			hn[hn_size-h0_size+j] -= (i-1)*h0[j];
		for (size_t j=0; j<h1_size; j++)
			h0[j] = h1[j];
		h0_size = h1_size;
		for (size_t j=0; j<h1_size+1; j++) 
			h1[j] = hn[j];
		h1_size++;
	}
	std::reverse(&hn[0], &hn[hn.order()]);
	return hn;
}

/// Factorial function.
template<class T>
T fact(T x, T p=1) {
	T m = 1;
	while (x > 1) {
		m *= x;
		x -= p;
	}
	return m;
}

template<class T>
void del(T** a, int n) {
	for (int i=0; i<n; i++)
		delete[] a[i];
	delete[] a;
}

// Ìåòîä íàèìåíüøèõ êâàäðàòîâ (îáùèé âèä)
// N - èñõîäíîå êîë-âî íåèçâåñòíûõ
// n - êîë-âî çíà÷èìûõ íåèçâåñòíûõ
template<class T>
void least_squares(T** P, const T* p, T** A, T* b, int n, int N)
{
	for (int k=0; k<n; k++) { // ïî íåèçâåñòíûì
		b[k] = 0.0;
		for (int j=0; j<n; j++) A[k][j] = 0.0;
		for (int i=0; i<N; i++) { // ïî ñòðîêàì
			for (int j=0; j<n; j++) {
				A[k][j] += P[i][k]*P[i][j];
//				cout << "A["<<k<<"]["<<j<<"] += P["<<i<<"]["<<k<<"]*P["<<i<<"]["<<j<<"]" << endl;
			}
			b[k] += P[i][k]*p[i];
//			cout << "b["<<k<<"] += P["<<i<<"]["<<k<<"]*p["<<i<<"]" << endl;
		}
	}
}

/// Cholesky decomposition.
template<class T>
void cholesky(T** A, T* b, int n, T* x)
{
	// íèæíÿÿ òðåóãîëüíàÿ ìàòðèöà
	T** L = A;
	// ðàçëîæåíèå A=L*T (T - òðàíñïîíèðîâàííîå L)
	for (int j=0; j<n; j++) {
		T sum = 0.0;
		for (int k=0; k<j; k++) {
			sum += L[j][k]*L[j][k];
		}
		L[j][j] = sqrt(A[j][j]-sum);
		for (int i=j+1; i<n; i++) {
			sum = 0.0;
			for (int k=0; k<j; k++) {
				sum += L[i][k]*L[j][k];
			}
			L[i][j] = (A[i][j]-sum)/L[j][j];
		}
	}
	//print(L, 0, n, "L");
	// ðåøåíèå L*y=b
	T* y = x;
	for (int i=0; i<n; i++) {
		T sum = 0.0;
		for (int j=0; j<i; j++) {
			sum += L[i][j]*y[j];
		}
		y[i] = (b[i] - sum)/L[i][i];
	}
	//print(0, y, n, "y");
	// ðåøåíèå T*x=y
	for (int i=n-1; i>=0; i--) {
		T sum = 0.0;
		for (int j=i+1; j<n; j++) {
			sum += L[j][i]*x[j];
		}
		x[i] = (y[i] - sum)/L[i][i];
	}
	//print(0, x, n, "x");
	// ïðîâåðêà
	/*for (int i=0; i<n; i++) {
		T sum = 0.0;
		for (int j=0; j<n; j++) {
			sum += x[j]*L[j][i];
		}
		sum -= y[i];
		cout << setw(5) << sum << endl;
	}*/
}

/// Implementation of least squares interpolation.
/// @a -- interpolation coefficients
template<class T>
void interpolate(const T* x,
				 const T* y,
				 int N, T* a,
				 int n)
{
	T** A = new T*[N];
	for (int i=0; i<N; i++)
		A[i] = new T[n];

	for (int i=0; i<N; i++)
		for (int k=0; k<n; k++)
			A[i][k] = pow(x[i], k);

	T* b2 = new T[n];
	T** A2 = new T*[n];
	for (int i=0; i<n; i++)
		A2[i] = new T[n];

	least_squares(A, y, A2, b2, n, N);
	cholesky(A2, b2, n, a);

	del(A, N);
	del(A2, n);
	delete[] b2;
}

/// Implementation of bisection method to find root of a function @func
/// in interval [@a,@b].
template <class T, class F>
T bisection(T a, T b, F func, T eps, uint max_iter=30)
{
	T c, fc;
	uint i = 0;
	do {
		c = T(0.5)*(a + b);
		fc = func(c);
		if (func(a)*fc < T(0)) b = c;
		if (func(b)*fc < T(0)) a = c;
		i++;
	} while (i<max_iter && (b-a) > eps && fabs(fc) > eps);
	return c;
}

}

#endif // EMATH_HH
