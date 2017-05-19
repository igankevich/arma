#ifndef ARMA_MODEL_HH
#define ARMA_MODEL_HH

#include "types.hh"
#include "ar_model.hh"
#include "ma_model.hh"

namespace arma {

	namespace generator {

		/**
		\brief Uses autoregressive moving average process (WIP).
		*/
		template <class T>
		struct ARMA_model: public AR_model<T>,
		                   public MA_model<T> {

			typedef AR_model<T> ar_model;
			typedef MA_model<T> ma_model;

			ARMA_model() = default;

			inline explicit
			ARMA_model(Array3D<T> acf, Shape3D ar_order, Shape3D ma_order):
			ar_model(slice_front(acf, ar_order).copy(), ar_order),
			ma_model(slice_back(acf, ma_order).copy(), ma_order),
			_acf_orig(acf)
			{}

			inline Array3D<T>
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
			operator>>(std::istream& in, ARMA_model<X>& rhs);

			void
			determine_coefficients() override;

		protected:
			T
			white_noise_variance(Array3D<T> phi, Array3D<T> theta) const;

		private:
			inline static Array3D<T>
			slice_back(Array3D<T> arr, Shape3D amount) {
				const Shape3D last = arr.shape() - 1;
				return arr(blitz::RectDomain<3>(arr.shape() - amount, last));
			}

			inline static Array3D<T>
			slice_front(Array3D<T> arr, Shape3D amount) {
				return arr(blitz::RectDomain<3>(Shape3D(0, 0, 0), amount - 1));
			}

			Array3D<T> _acf_orig;
		};

		template <class T>
		std::istream&
		operator>>(std::istream& in, ARMA_model<T>& rhs);

	}

}

#endif // ARMA_MODEL_HH
