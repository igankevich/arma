#ifndef STATS_SUMMARY_HH
#define STATS_SUMMARY_HH

#include <cmath>
#include <string>

#include <blitz/array.h>

#include "qq_graph.hh"

namespace arma {

	namespace stats {

		/**
		   \brief A summary of estimated statistical properties of an array
		   with their theoretical counterparts.

		   Includes
		   - estimated and theoretical mean and variance,
		   - quantile-quantile plot of data distribution against
		   theoretical distribution.
		 */
		template <class T>
		struct Summary {

			template <int N, class D>
			Summary(
				blitz::Array<T, N> rhs,
				T m,
				T var,
				D dist,
				std::string name,
				bool needsvariance,
				T tolerance=T(0.1)
			):
			_expected_mean(m),
			_mean(stats::mean(rhs)),
			_expected_variance(var),
			_variance(needsvariance ? stats::variance(rhs) : T(0)),
			_graph(dist, rhs),
			_name(name),
			_needvariance(needsvariance),
			_tolerance(tolerance)
			{}

			inline T
			qdistance() const {
				return this->_graph.distance();
			}

			void
			write_quantile_graph();

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const Summary<X>& rhs);

			static void
			print_header(std::ostream& out);

			inline const std::string&
			name() const noexcept {
				return this->_name;
			}

			inline bool
			has_variance() const noexcept {
				return this->_needvariance;
			}

			inline T
			mean() const noexcept {
				return this->_mean;
			}

			inline T
			expected_mean() const noexcept {
				return this->_expected_mean;
			}

			inline T
			variance() const noexcept {
				return this->_variance;
			}

			inline T
			expected_variance() const noexcept {
				return this->_expected_variance;
			}

			inline bool
			mean_ok() const noexcept {
				return !std::isfinite(this->_expected_mean) ||
				       std::abs(this->_expected_mean - this->_mean)
				       < this->_tolerance;
			}

			inline bool
			variance_ok() const noexcept {
				return !std::isfinite(this->_expected_variance) ||
				       std::abs(this->_expected_variance - this->_variance)
				       < this->_tolerance;
			}

			inline bool
			qdistance_ok() const {
				return this->qdistance() < T(0.06);
			}

		private:
			T _expected_mean;
			T _mean;
			T _expected_variance;
			T _variance;
			stats::QQ_graph<T> _graph;
			std::string _name;
			bool _needvariance = true;
			T _tolerance=T(0.1);
		};

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const Summary<T>& rhs);

		template <class T, int N, class D>
		Summary<T>
		make_summary(
			blitz::Array<T, N> rhs,
			T m,
			T var,
			D dist,
			std::string name,
			T tol
		) {
			return Summary<T>(rhs, m, var, dist, name, true, tol);
		}

		template <class T, int N, class D>
		Summary<T>
		make_summary(
			blitz::Array<T, N> rhs,
			T m,
			D dist,
			std::string name,
			T
			tol
		) {
			return Summary<T>(rhs, m, T(0), dist, name, false, tol);
		}

	}

}

#endif // STATS_SUMMARY_HH
