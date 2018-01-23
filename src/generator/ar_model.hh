#ifndef AR_MODEL_HH
#define AR_MODEL_HH

#include "ar_algorithm.hh"
#include "arma.hh"
#include "basic_arma_model.hh"
#include "discrete_function.hh"
#include "types.hh"

/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.


#if ARMA_BSCHEDULER
namespace {
	template <class T>
	class ar_partition_kernel;
	template <class T>
	class ar_master_kernel;
}
#endif

namespace arma {

	namespace generator {

		/**
		   \brief Uses autoregressive process, standing waves.
		   \ingroup generators
		 */
		template <class T>
		class AR_model: public Basic_ARMA_model<T> {

		private:
			/// The size of partitions that are computed in parallel.
			Shape3D _partition = Shape3D(0,0,0);
			/// AR coefficients.
			Array3D<T> _phi;
			/// The algorithm for determining the coefficients.
			AR_algorithm _algorithm = AR_algorithm::Choi;
			/// White noise variance, obtained during coefficient determination.
			T _varwn = T(0);

		public:
			typedef Discrete_function<T,3> acf_type;

			AR_model() = default;

			inline explicit
			AR_model(acf_type acf, Shape3D order):
			Basic_ARMA_model<T>(acf, order),
			_phi(order)
			{}

			inline Array3D<T>
			coefficients() const {
				return this->_phi;
			}

			inline T
			white_noise_variance() const override {
				return this->_varwn;
			}

			inline bool
			writes_in_parallel() const noexcept override {
				return this->oflags().isset(Output_flags::Binary);
			}

			void
			validate() const override;

			Array3D<T>
			do_generate() override;

			void
			determine_coefficients() override;

			#if ARMA_BSCHEDULER
			void
			act() override;

			void
			react(bsc::kernel* child) override;

			void
			write(sys::pstream& out) const override;

			void
			read(sys::pstream& in) override;

			template <class X>
			friend class ::ar_partition_kernel;

			template <class X>
			friend class ::ar_master_kernel;
			#endif

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const AR_model<X>& rhs);

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, AR_model<X>& rhs);

		protected:
			void
			generate_surface(Array3D<T>& zeta, const Domain3D& subdomain);

			T
			white_noise_variance(Array3D<T> phi) const;

			void
			write(std::ostream& out) const override;

			void
			read(std::istream& in) override;

		private:

			/// Determine coefficients by simple Gauss elimintation.
			void
			determine_coefficients_gauss();

			/// Choi recursive-order algorithm \cite Choi1999.
			void
			determine_coefficients_choi();

		};

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const AR_model<T>& rhs) {
			rhs.AR_model<T>::write(out);
			return out;
		}

		template <class T>
		std::istream&
		operator>>(std::istream& in, AR_model<T>& rhs) {
			rhs.AR_model<T>::read(in);
			return in;
		}

	}

}

#endif // AR_MODEL_HH
