#ifndef GENERATOR_MODEL_HH
#define GENERATOR_MODEL_HH

#include "types.hh"
#include "grid.hh"
#include <istream>
#include <ostream>

namespace arma {

	/// \brief Wavy surface generators.
	namespace generator {

		/// \brief A base class for ARMA generators.
		template<class T>
		class Basic_model {

			typedef Grid<T,3> grid_type;

		protected:
			/// Wavy surface grid.
			grid_type _outgrid;

			virtual void write(std::ostream& out) const {}
			virtual void read(std::istream& in) {}

		public:

			Basic_model() = default;
			Basic_model(const Basic_model&) = default;
			Basic_model(Basic_model&&) = default;
			virtual ~Basic_model() = default;

			inline void
			setgrid(const grid_type& rhs) noexcept {
				this->_outgrid = rhs;
			}

			const grid_type&
			grid() const noexcept {
				return this->_outgrid;
			}

			virtual T white_noise_variance() const = 0;
			virtual void validate() const = 0;
			virtual void determine_coefficients() = 0;
			virtual Array3D<T> generate() = 0;

			virtual void
			operator()(
				Array3D<T>& zeta,
				Array3D<T>& eps,
				const Domain3D& subdomain
			) = 0;

			inline void
			operator()(Array3D<T>& zeta, Array3D<T>& eps) {
				operator()(zeta, eps, zeta.domain());
			}

			inline friend std::ostream&
			operator<<(std::ostream& out, const Basic_model& rhs) {
				rhs.write(out);
				return out;
			}

			inline friend std::istream&
			operator>>(std::istream& in, Basic_model& rhs) {
				rhs.read(in);
				return in;
			}

		};

	}

}

#endif // GENERATOR_MODEL_HH
