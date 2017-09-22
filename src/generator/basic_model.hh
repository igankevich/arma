#ifndef GENERATOR_MODEL_HH
#define GENERATOR_MODEL_HH

#include "types.hh"
#include "grid.hh"
#include "output_flags.hh"
#include "parallel_mt.hh"
#include <istream>
#include <ostream>

namespace arma {

	/// \brief Wavy surface generators.
	namespace generator {

		/**
		\defgroup generators Wavy surface generation
		\brief Generators based on ocean wavy surface simulation models.
		*/

		/**
		\brief A base class for all generators.
		\ingroup generators
		*/
		template<class T>
		class Basic_model {

		public:
			typedef Grid<T,3> grid_type;

		protected:
			/// Wavy surface grid.
			grid_type _outgrid;
			Output_flags _oflags;
			/// Whether seed PRNG or not. This flag is needed for
			/// reproducible tests.
			bool _noseed = false;

			virtual void write(std::ostream& out) const {}
			virtual void read(std::istream& in) {}

			inline prng::clock_type::rep
			newseed() noexcept {
				return this->_noseed
					? prng::clock_type::rep(0)
					: prng::clock_seed();
			}

		public:

			Basic_model() = default;
			Basic_model(const Basic_model&) = default;
			Basic_model(Basic_model&&) = default;
			virtual ~Basic_model() = default;

			inline void
			setgrid(const grid_type& rhs) noexcept {
				this->_outgrid = rhs;
			}

			inline const grid_type&
			grid() const noexcept {
				return this->_outgrid;
			}

			inline Output_flags
			oflags() const noexcept {
				return this->_oflags;
			}

			virtual bool
			writes_in_parallel() const noexcept {
				return false;
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
