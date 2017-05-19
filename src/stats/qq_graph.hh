#ifndef STATS_QQ_GRAPH_HH
#define STATS_QQ_GRAPH_HH

#include <blitz/array.h>
#include <algorithm>

#include "statistics.hh"

namespace arma {

	namespace stats {

		/// \brief Q-Q plot of a discretely given function.
		template <class T>
		struct QQ_graph {

			template <int N, class D>
			QQ_graph(D dist, blitz::Array<T, N> rhs, size_t nquantiles = 100):
			_expected(nquantiles),
			_real(nquantiles)
			{
				blitz::Array<T, N> data = rhs.copy();
				/// 1. Calculate expected quantile values from supplied quantile
				/// function.
				for (size_t i = 0; i < nquantiles; ++i) {
					const T f = T(1.0) / T(nquantiles - 1) * T(i);
					_expected(i) = dist.quantile(f);
				}
				/// 2. Calculate real quantiles from data.
				std::sort(data.data(), data.data() + data.numElements());
				for (size_t i = 0; i < nquantiles; ++i) {
					const T f = T(1.0) / T(nquantiles - 1) * T(i);
					_real(i) = quantile(data, f);
				}
			}

			/// Calculate distance between two quantile vectors.
			T
			distance() const;

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const QQ_graph<X>& rhs);

		private:
			blitz::Array<T, 1> _expected;
			blitz::Array<T, 1> _real;
		};

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const QQ_graph<T>& rhs);

	}

}

#endif // STATS_QQ_GRAPH_HH
