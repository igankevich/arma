#ifndef GENERATOR_MODEL_HH
#define GENERATOR_MODEL_HH

#include "types.hh"
#include "grid.hh"
#include "output_flags.hh"
#include <istream>
#include <ostream>

namespace arma {

	/// \brief Wavy surface generators.
	namespace generator {

		/// \brief A base class for ARMA generators.
		template<class T>
		class Basic_model {

		public:
			typedef Grid<T,3> grid_type;

		protected:
			/// Wavy surface grid.
			grid_type _outgrid;
			Output_flags _oflags;

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

			Output_flags
			vscheme() const noexcept {
				return this->_oflags;
			}

			virtual void validate() const {}
			virtual Array3D<T> generate() = 0;
			virtual void verify(Array3D<T> zeta) const {}

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
