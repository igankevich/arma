#ifndef STATS_SUMMARY_HH
#define STATS_SUMMARY_HH

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
			Summary(blitz::Array<T, N> rhs, T m, T var, D dist, std::string name):
			_expected_mean(m),
			_mean(stats::mean(rhs)),
			_expected_variance(var),
			_variance(stats::variance(rhs)),
			_graph(dist, rhs),
			_name(name)
			{}

			inline T
			qdistance() const {
				return _graph.distance();
			}

			void
			write_quantile_graph();

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const Summary<X>& rhs);

			static void
			print_header(std::ostream& out);

		private:
			T _expected_mean;
			T _mean;
			T _expected_variance;
			T _variance;
			stats::QQ_graph<T> _graph;
			std::string _name;
		};

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const Summary<T>& rhs);

		template <class T, int N, class D>
		Summary<T>
		make_summary(blitz::Array<T, N> rhs, T m, T var, D dist, std::string name) {
			return Summary<T>(rhs, m, var, dist, name);
		}

	}

}

#endif // STATS_SUMMARY_HH
