#ifndef NONLINEAR_TRANSFORMS_HH
#define NONLINEAR_TRANSFORMS_HH

#include <utility>
#include "equations.hh"
#include "linalg.hh"

namespace arma {

	namespace nonlinear {

		/**
		\brief Transforms cumulative distribution function from old_dist to
		new_dist at specified grid points.
		\date 2017-05-20
		\author Ivan Gankevich

		\param[in] grid defines points in which transformation is done
		\param[in] old_dist current cumulative distribution function
		\param[in] new_dist desired cumulative distribution function
		\param[in] solver equation solver

		\return a pair where the first element contains grid points in which
		distribution was transformed, and the second element contains new
		distribution function values.

		\see arma::stats::Gaussian
		\see linalg::Bisection
		*/
		template <class T, class Dist1, class Dist2, class Solver>
		std::pair<blitz::Array<T,1>,blitz::Array<T,1>>
		transform_CDF(
			const Domain<T, 1>& grid,
			Dist1 old_dist,
			Dist2 new_dist,
			Solver solver
		) {
			const int n = grid.num_points();
			blitz::Array<T,1> x(n), y(n);
			for (int i=0; i<n; ++i) {
				const int xi = grid(i);
				x(i) = xi;
				y(i) = solver(
					Equation_CDF<T, Dist2>(new_dist, old_dist.cdf(xi))
				);
			}
			return std::make_pair(x, y);
		}

		/**
		\brief Transforms each data point from old to new distribution.
		\date 2017-05-20
		\author Ivan Gankevich
		*/
		template <class T, class Dist1, class Dist2, class Solver>
		void
		transform_data(
			const T* data,
			const int n,
			Dist1 old_dist,
			Dist2 new_dist,
			Solver solver
		) {
			for (int i=0; i<n; ++i) {
				data[i] = solver(
					Equation_CDF<T, Dist2>(new_dist, old_dist.cdf(data[i]))
				);
			}
		}

		/**
		\brief Transform ACF to the distribution function expanded into
		       Gram--Charlier series.
		\date 2017-05-20
		\author Ivan Gankevich
		*/
		template <class T, class Solver>
		void
		transform_ACF(
			const T* data,
			const int n,
			const blitz::Array<T,1> gram_charlier_coefs,
			Solver solver
		) {
			for (int i=0; i<n; ++i) {
				data[i] = solver(
					Equation_ACF<T>(gram_charlier_coefs, data[i])
				);
			}
		}

	}

}

#endif // NONLINEAR_TRANSFORMS_HH

