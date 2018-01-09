#ifndef LINALG_HH
#define LINALG_HH

#include <cassert>
#include <blitz/array.h>
#include <lapacke/lapacke.h>
#include <openblas/cblas.h>
#include "apmath/closed_interval.hh"

/**
\brief Linear algebra subroutines.
\author Ivan Gankevich
\date 2016-07-26
*/
namespace linalg {

	using ::arma::apmath::closed_interval;

	template <class T>
	using Matrix = blitz::Array<T, 2>;
	template <class T>
	using Vector = blitz::Array<T, 1>;

	/**
	\brief Solve linear system \f$A x=b\f$ via Cholesky decomposition.
	\param[in]    A input matrix (lhs).
	\param[in,out] b input and output vector (rhs).
	*/
	template <class T>
	void
	cholesky(Matrix<T>& A, Vector<T>& b);

	template <class T>
	void
	inverse(Matrix<T>& A);

	template <int N>
	blitz::Array<float,N>
	operator*(Matrix<float> lhs, blitz::Array<float,N> rhs) {
		const int m = lhs.rows(), n = lhs.cols();
		blitz::Array<float,N> result(rhs.shape());
		result = 0;
		cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.f, lhs.data(), m,
		     rhs.data(), 1, 0.f, result.data(), 1);
		return result;
	}

	template <int N>
	blitz::Array<double,N>
	operator*(Matrix<double> lhs, blitz::Array<double,N> rhs) {
		const int m = lhs.rows(), n = lhs.cols();
		blitz::Array<double,N> result(rhs.shape());
		result = 0;
		cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, lhs.data(), m,
		     rhs.data(), 1, 0.0, result.data(), 1);
		return result;
	}

	template <int N>
	double
	dot(blitz::Array<double,N> lhs, blitz::Array<double,N> rhs) {
		assert(lhs.numElements() == rhs.numElements());
		const int n = lhs.numElements();
		return cblas_ddot(n, lhs.data(), 1, rhs.data(), 1);
	}

	template <class T>
	bool
	is_symmetric(Matrix<T>& rhs);

	/**
	\brief Use Cholesky decomposition to determine
	if matrix \p lhs is positive definite.
	*/
	template <class T>
	bool
	is_positive_definite(Matrix<T>& rhs);

	template <class T>
	bool
	is_toeplitz(Matrix<T>& rhs);

	/**
	\brief Make matrix square via least squares.

	\param[in] P input matrix (lhs).
	\param[in] p input vector (rhs).
	\param[out] A output matrix (lhs).
	\param[out] b output vector (rhs).
	*/
	template <class T>
	void
	least_squares(
		const Matrix<T>& P,
		const Vector<T>& p,
		Matrix<T>& A,
		Vector<T>& b
	);

	/**
	\brief Least squares interpolation.

	\param  x,y interpolation nodes.
	\param  n   no. of interpolation coefficients.
	\return     interpolation coefficients.
	*/
	template <class T>
	Vector<T>
	interpolate(const Vector<T>& x, const Vector<T>& y, const int n);

	/**
	\brief Solve equation \f$f(x)=0\f$ via bisection method.
	\param a left margin (\f$x_0\f$).
	\param b right margin (\f$x_1\f$).
	\param func lhs of equation (\f$f(x)\f$).
	\return solution to \f$f(x)=0\f$.
	*/
	template <class T, class F>
	T
	bisection(T a, T b, F func, T eps, int max_iter = 30) {
		T c, fc;
		int i = 0;
		do {
			c = T(0.5) * (a + b);
			fc = func(c);
			if (func(a) * fc < T(0)) b = c;
			if (func(b) * fc < T(0)) a = c;
			i++;
		} while (i < max_iter && (b - a) > eps && std::abs(fc) > eps);
		// TODO eps is used as absolute error and as a minimal interval size
		return c;
	}

	/**
	\brief Functor for bisection algorithm.
	\date 2017-05-20
	\author Ivan Gankevich
	*/
	template <class T>
	class Bisection {
		closed_interval<T> _interval;
		T _eps;
		int _niterations;

	public:
		inline
		Bisection(T a, T b, T eps, int niterations) noexcept:
		_interval(a, b),
		_eps(eps),
		_niterations(niterations)
		{}

		template <class Func>
		T
		operator()(Func func) const noexcept {
			return bisection<T>(
				this->_interval.first(),
				this->_interval.last(),
				func,
				this->_eps,
				this->_niterations
			);
		}

		inline void
		interval(T a, T b) noexcept {
			this->_interval = closed_interval<T>(a, b);
		}

		inline const closed_interval<T>&
		interval() const noexcept {
			return this->_interval;
		}

		inline int
		num_iterations() const noexcept {
			return this->_niterations;
		}

		template <class X>
		friend std::istream&
		operator>>(std::istream& in, Bisection<X>& rhs);

		template <class X>
		friend std::ostream&
		operator<<(std::ostream& out, const Bisection<X>& rhs);
	};

	template <class T>
	std::istream&
	operator>>(std::istream& in, Bisection<T>& rhs);

	template <class T>
	std::ostream&
	operator<<(std::ostream& out, const Bisection<T>& rhs);

}

#endif // LINALG_HH
