#ifndef NONLINEAR_NIT_TRANSFORM_HH
#define NONLINEAR_NIT_TRANSFORM_HH

#include <istream>
#include <ostream>

#include "stats/gaussian.hh"
#include "stats/skew_normal.hh"
#include "stats/gram_charlier.hh"
#include "linalg.hh"
#include "types.hh"
#include "domain.hh"

namespace arma {

	namespace nonlinear {

		namespace bits {

			enum struct Distribution {
				Gram_Charlier = 0,
				Skew_normal = 1
			};

			std::istream&
			operator>>(std::istream& in, Distribution& rhs);

			std::ostream&
			operator<<(std::ostream& out, const Distribution& rhs);

			const char*
			to_string(Distribution rhs);

		}

		/**
		\brief Non-linear inertialess tranformation (NIT).
		\date 2017-06-03
		\author Ivan Gankevich

		Transforms wavy surface ditribution law to a specified one. Transforms
		ACF to preserve spectrum of of ARMA process.
		*/
		template <class T>
		class NIT_transform {

			typedef stats::Gaussian<T> normaldist_type;
			typedef stats::Skew_normal<T> skewnormaldist_type;
			typedef stats::Gram_Charlier<T> gramcharlierdist_type;
			typedef linalg::Bisection<T> solver_type;

			static const unsigned int default_interpolation_order = 12;
			static const unsigned int default_gram_charlier_order = 10;
			static constexpr const T default_absolute_error = T(1e-6);
			static const unsigned int default_niterations = 100;

			bits::Distribution _targetdist = bits::Distribution::Gram_Charlier;
			skewnormaldist_type  _skewnormal;
			gramcharlierdist_type  _gramcharlier;
			unsigned int _intnodes = 100;
			unsigned int _maxintorder = default_interpolation_order;
			unsigned int _maxexpansionorder = default_gram_charlier_order;
			solver_type _cdfsolver, _acfsolver;
			Array1D<T> _xnodes, _ynodes;

		public:
			NIT_transform():
			_cdfsolver(T(0), T(1), default_absolute_error, default_niterations),
			_acfsolver(T(0), T(1), default_absolute_error, default_niterations)
			{}

			void
			transform_ACF(Array3D<T>& acf);

			void
			transform_realisation(Array3D<T> acf, Array3D<T>& realisation);

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const NIT_transform<X>& rhs);

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, NIT_transform<X>& rhs);

		private:
			void
			transform_CDF(Array3D<T> acf);

			std::pair<arma::Array1D<T>,arma::Array1D<T>>
			do_transform_CDF(const T stdev, const Domain<T,1>& grid);

			template <class Dist>
			void
			do_transform_realisation(
				Array3D<T> acf,
				Array3D<T>& realisation,
				Dist& dist
			);

			void
			read_dist(std::istream& in);

		};

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const NIT_transform<T>& rhs);

		template <class T>
		std::istream&
		operator>>(std::istream& in, NIT_transform<T>& rhs);

	}

}

#endif // NONLINEAR_NIT_TRANSFORM_HH
