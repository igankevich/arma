#ifndef ARMA_MODEL_HH
#define ARMA_MODEL_HH

#include "types.hh"
#include "ar_generator.hh"
#include "ma_generator.hh"
#include "discrete_function.hh"

namespace arma {

	namespace generator {

		/**
		\brief Uses autoregressive moving average process (WIP).
		*/
		template <class T>
		struct ARMA_generator: public AR_generator<T>,
		                   public MA_generator<T> {

			typedef AR_generator<T> ar_model;
			typedef MA_generator<T> ma_model;
			typedef Discrete_function<T,3> acf_type;

			ARMA_generator() = default;

			inline explicit
			ARMA_generator(acf_type acf, Shape3D ar_order, Shape3D ma_order):
			ar_model(slice_front(acf, ar_order), ar_order),
			ma_model(slice_back(acf, ma_order), ma_order),
			_acf_orig(acf)
			{}

			inline acf_type
			acf() const {
				return _acf_orig;
			}

			inline Shape3D
			order() const {
				return ar_model::order() + ma_model::order();
			}

			T
			white_noise_variance() const override;

			void
			validate() const override;

			inline void
			operator()(Array3D<T>& zeta, Array3D<T>& eps) {
				operator()(zeta, eps, zeta.domain());
			}

			void
			operator()(
				Array3D<T>& zeta,
				Array3D<T>& eps,
				const Domain3D& subdomain
			) override;

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, ARMA_generator<X>& rhs);

			void
			determine_coefficients() override;

		protected:
			T
			white_noise_variance(Array3D<T> phi, Array3D<T> theta) const;

		private:
			inline static acf_type
			slice_back(acf_type arr, Shape3D amount) {
				const Shape3D last = arr.shape() - 1;
				Array3D<T> res = arr(
					blitz::RectDomain<3>(arr.shape() - amount, last)
				);
				acf_type result;
				result.reference(res);
				result.setgrid(arr.grid());
				return result;
			}

			inline static acf_type
			slice_front(acf_type arr, Shape3D amount) {
				Array3D<T> res = arr(
					blitz::RectDomain<3>(Shape3D(0, 0, 0), amount - 1)
				);
				acf_type result;
				result.reference(res);
				result.setgrid(arr.grid());
				return result;
			}

			acf_type _acf_orig;
		};

		template <class T>
		std::istream&
		operator>>(std::istream& in, ARMA_generator<T>& rhs);

	}

}

#endif // ARMA_MODEL_HH
