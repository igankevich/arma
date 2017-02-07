#ifndef ARMA_MODEL_HH
#define ARMA_MODEL_HH

#include "types.hh" // for size3, Array3D, Array2D, Array1D, Zeta

namespace arma {

	template <class T>
	struct ARMA_model : public Autoregressive_model<T>,
	                    public Moving_average_model<T> {

		typedef Autoregressive_model<T> ar_model;
		typedef Moving_average_model<T> ma_model;

		ARMA_model(Array3D<T> acf, size3 ar_order, size3 ma_order)
		    : Autoregressive_model<T>(slice_front(acf, ar_order).copy(), ar_order),
		      Moving_average_model<T>(slice_back(acf, ma_order).copy(), ma_order),
		      _acf_orig(acf) {}

		ACF<T>
		acf() const {
			return _acf_orig;
		}

		size3
		order() const {
			return ar_model::order() + ma_model::order();
		}

		T
		white_noise_variance(Array3D<T> phi, Array3D<T> theta) const {
			return ar_model::white_noise_variance(phi) *
			       ma_model::white_noise_variance(theta) /
			       ar_model::acf_variance();
		}

		T
		white_noise_variance() const {
			return white_noise_variance(ar_model::coefficients(),
			                            ma_model::coefficients());
		}

		void
		validate() const {
			ar_model::validate();
			std::clog << "AR process is OK." << std::endl;
			ma_model::validate();
		}

		void
		operator()(Array3D<T>& zeta, Array3D<T>& eps) {
			operator()(zeta, eps, zeta.domain());
		}

		void
		operator()(
			Array3D<T>& zeta,
			Array3D<T>& eps,
			const Domain3D& subdomain
		) {
			ma_model::operator()(zeta, eps, subdomain);
			ar_model::operator()(zeta, zeta, subdomain);
		}

		template<class Options>
		void
		determine_coefficients(Options opts) {
			using namespace blitz;
			if (product(ar_model::order()) > 0) {
				ar_model::determine_coefficients(opts);
			}
//			ma_model::recompute_acf(_acf_orig, ar_model::coefficients());
			ma_model::determine_coefficients(opts);
		}

	private:
		static Array3D<T>
		slice_back(Array3D<T> arr, size3 amount) {
			const size3 last = arr.shape() - 1;
			return arr(blitz::RectDomain<3>(arr.shape() - amount, last));
		}

		static Array3D<T>
		slice_front(Array3D<T> arr, size3 amount) {
			return arr(blitz::RectDomain<3>(size3(0, 0, 0), amount - 1));
		}

		Array3D<T> _acf_orig;
	};
}

#endif // ARMA_MODEL_HH
