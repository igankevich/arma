#include "qq_graph.hh"
#include <cmath>

namespace {

	template <class X>
	struct QPair {

		QPair(X a, X b) : first(a), second(b) {}

		friend std::ostream&
		operator<<(std::ostream& out, const QPair& rhs) {
			return out << std::setw(20) << rhs.first << std::setw(20)
					   << rhs.second;
		}

		X first;
		X second;
	};

	template <class T, int N>
	T
	abs_max(blitz::Array<T, N> rhs) {
		return std::max(std::abs(blitz::max(rhs)), std::abs(blitz::min(rhs)));
	}


}

template<class T>
T
arma::stats::QQ_graph<T>::distance() const {
	using blitz::sum;
	using blitz::abs;
	/// 1. Omit first and last quantiles if they are not finite in
	/// expected distribution.
	int x0 = !std::isfinite(_expected(0)) ? 1 : 0;
	int x1 = !std::isfinite(_expected(_expected.size() - 1))
				 ? _expected.size() - 2
				 : _expected.size() - 1;
	blitz::Range r(x0, x1);
	/// 2. Rescale all values to \f$[0,1]\f$ range.
	const T scale = std::max(abs_max(_expected(r)), abs_max(_real(r)));
	const int nquantiles = x1 - x0 + 1;
	return sum(abs(_expected(r) - _real(r))) / scale / T(nquantiles);
}

template <class T>
std::ostream&
arma::stats::operator<<(std::ostream& out, const QQ_graph<T>& rhs) {
	std::transform(
		rhs._expected.begin(), rhs._expected.end(), rhs._real.begin(),
		std::ostream_iterator<QPair<T>>(out, "\n"),
		[](float lhs, float rhs) { return QPair<T>(lhs, rhs); });
	return out;
}


template class arma::stats::QQ_graph<ARMA_REAL_TYPE>;

template std::ostream&
arma::stats::operator<<(
	std::ostream& out,
	const QQ_graph<ARMA_REAL_TYPE>& rhs
);
