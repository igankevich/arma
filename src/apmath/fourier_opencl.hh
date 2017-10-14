#ifndef APMATH_FOURIER_OPENCL_HH
#define APMATH_FOURIER_OPENCL_HH

#include <ostream>
#include <complex>
#include <type_traits>
#include <blitz/array.h>
#include <clFFT.h>

#include "opencl/opencl.hh"
#include "fourier_direction.hh"

namespace arma {

	namespace apmath {

		template <class T, int N>
		class Fourier_workspace {

		public:
			typedef blitz::TinyVector<int,N> shape_type;

		private:
			shape_type _shape;

		public:
			Fourier_workspace() = default;
			Fourier_workspace(Fourier_workspace&&) = default;
			Fourier_workspace(const Fourier_workspace&) = delete;

			Fourier_workspace&
			operator=(const Fourier_workspace&) = delete;

			inline explicit
			Fourier_workspace(const shape_type& shp):
			_shape(shp) {}

			inline shape_type
			shape() const noexcept {
				return this->_shape;
			}

		};

		template <class T, int N>
		class Fourier_transform {

			static_assert(N > 0, "bad no. of dimensions");
			static_assert(N <= 3, "bad no. of dimensions");
			static_assert(
				std::is_same<T,std::complex<float>>::value ||
				std::is_same<T,std::complex<double>>::value,
				"bad types"
			);

		public:
			typedef Fourier_workspace<T,N> workspace_type;
			typedef blitz::TinyVector<int,N> shape_type;
			typedef cl::Buffer buffer_type;
			typedef blitz::Array<T,N> array_type;

		private:
			shape_type _shape;
			clfftSetupData _fft;
			clfftPlanHandle _fftplan;

		public:
			Fourier_transform();
			~Fourier_transform();
			Fourier_transform(Fourier_transform&&) = default;
			Fourier_transform(const Fourier_transform&) = delete;

			Fourier_transform&
			operator=(const Fourier_transform&) = delete;

			inline explicit
			Fourier_transform(const shape_type& shape) {
				init(shape);
			}

			void
			init(const shape_type& shp);

			inline shape_type
			shape() const noexcept {
				return this->_shape;
			}

			inline static int
			dimensions() noexcept {
				return N;
			}

			inline array_type
			forward(array_type rhs) {
				workspace_type ws(this->new_workspace());
				return transform(rhs, ws, Fourier_direction::Forward);
			}

			inline array_type
			forward(array_type rhs, workspace_type& workspace) {
				return transform(rhs, workspace, Fourier_direction::Forward);
			}

			inline array_type
			backward(array_type rhs) {
				workspace_type ws(this->new_workspace());
				return transform(rhs, ws, Fourier_direction::Backward);
			}

			inline array_type
			backward(array_type rhs, workspace_type& workspace) {
				return transform(rhs, workspace, Fourier_direction::Backward);
			}

			array_type
			transform(
				array_type rhs,
				workspace_type& workspace,
				Fourier_direction dir
			);

			inline buffer_type
			forward(buffer_type rhs) {
				workspace_type ws(this->new_workspace());
				return transform(rhs, ws, Fourier_direction::Forward);
			}

			inline buffer_type
			forward(buffer_type rhs, workspace_type& workspace) {
				return transform(rhs, workspace, Fourier_direction::Forward);
			}

			inline buffer_type
			backward(buffer_type rhs) {
				workspace_type ws(this->new_workspace());
				return transform(rhs, ws, Fourier_direction::Backward);
			}

			inline buffer_type
			backward(buffer_type rhs, workspace_type& workspace) {
				return transform(rhs, workspace, Fourier_direction::Backward);
			}

			buffer_type
			transform(
				buffer_type rhs,
				workspace_type& workspace,
				Fourier_direction dir
			);

			inline workspace_type
			new_workspace() const {
				return workspace_type(this->shape());
			}

			inline friend std::ostream&
			operator<<(std::ostream& out, const Fourier_transform& rhs) {
				return out << "shape=" << rhs.shape();
			}

		};

	}

}

#endif // APMATH_FOURIER_OPENCL_HH
