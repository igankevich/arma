#ifndef STATS_WEIBULL_HH
#define STATS_WEIBULL_HH

#include <istream>
#include <ostream>
#include <gsl/gsl_cdf.h>

namespace arma {

	namespace stats {

		/// \brief Weibull distribution.
		template <class T>
		struct Weibull {

			Weibull(T a, T b):
			_a(a),
			_b(b)
			{}

			T
			quantile(T f) {
				return gsl_cdf_weibull_Pinv(f, _a, _b);
			}

		private:
			T _a; //< lambda
			T _b; //< k
		};

	}

}

#endif // STATS_WEIBULL_HH
