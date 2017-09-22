#ifndef APMATH_FOURIER_GSL_HH
#define APMATH_FOURIER_GSL_HH

#include <complex>
#include <vector>

#include "apmath/fourier_direction.hh"
#include "bits/fourier_gsl.hh"
#include "types.hh"

namespace arma {

	namespace apmath {

		/**
		\brief An opaque object which holds per-thread state of
		\link Fourier_transform Fourier transform\endlink.
		*/
		template <class T, int N>
		class Fourier_workspace {

		public:
			typedef typename bits::Fourier_config<T>::workspace_type workspace_type;
			typedef blitz::TinyVector<int,N> shape_type;

		private:
			std::vector<workspace_type> _workspaces;

		public:
			Fourier_workspace() = default;
			Fourier_workspace(Fourier_workspace&&) = default;
			Fourier_workspace(const Fourier_workspace&) = delete;

			Fourier_workspace&
			operator=(const Fourier_workspace&) = delete;

			inline explicit
			Fourier_workspace(const shape_type& shp) {
				using blitz::any;
				for (int i=0; i<N; ++i) {
					this->_workspaces.emplace_back(shp(i));
				}
			}

			inline shape_type
			shape() const noexcept {
				shape_type result;
				const int n = this->_workspaces.size();
				for (int i = 0; i < n; ++i) {
					result(i) = this->_workspaces[i].size();
				}
				return result;
			}

			inline const workspace_type&
			operator[](int i) const noexcept {
				return this->_workspaces[i];
			}

			inline workspace_type&
			operator[](int i) noexcept {
				return this->_workspaces[i];
			}

		};

		/// \brief Multidimensional Fourier transform based on GSL library routines.
		template <class T, int N>
		class Fourier_transform {

		public:
			typedef typename bits::Fourier_config<T>::transform_type transform_type;
			typedef Fourier_workspace<T,N> workspace_type;
			typedef blitz::TinyVector<int,N> shape_type;
			typedef ::arma::Array<T,N> array_type;

		private:
			std::vector<transform_type> _transforms;

		public:
			Fourier_transform() = default;
			Fourier_transform(Fourier_transform&&) = default;
			Fourier_transform(const Fourier_transform&) = delete;

			Fourier_transform&
			operator=(const Fourier_transform&) = delete;

			inline explicit
			Fourier_transform(const shape_type& shape) {
				init(shape);
			}

			inline void
			init(const shape_type& shp) {
				if (blitz::any(shp != shape())) {
					this->_transforms.clear();
					for (int i = 0; i < N; ++i) {
						this->_transforms.emplace_back(shp(i));
					}
				}
			}

			inline shape_type
			shape() const noexcept {
				shape_type result;
				const int n = this->_transforms.size();
				for (int i = 0; i < n; ++i) {
					result(i) = this->_transforms[i].size();
				}
				return result;
			}

			inline static int
			dimensions() noexcept {
				return N;
			}

			inline array_type
			forward(array_type rhs) {
				workspace_type ws(this->new_workspace());
				return transform(rhs, ws, apmath::Fourier_direction::Forward);
			}

			inline array_type
			forward(array_type rhs, workspace_type& workspace) {
				return transform(rhs, workspace, apmath::Fourier_direction::Forward);
			}

			inline array_type
			backward(array_type rhs) {
				workspace_type ws(this->new_workspace());
				return transform(rhs, ws, apmath::Fourier_direction::Backward);
			}

			inline array_type
			backward(array_type rhs, workspace_type& workspace) {
				return transform(rhs, workspace, apmath::Fourier_direction::Backward);
			}

			inline array_type
			transform(
				array_type rhs,
				workspace_type& workspace,
				apmath::Fourier_direction dir
			) {
				const int n = this->_transforms.size();
				for (int i = 0; i < n; ++i) {
					const int stride = rhs.stride(i);
					const int extent = rhs.extent(i);
					const int block_size = extent*stride;
					const int nblocks = rhs.numElements() / block_size;
					/*
					std::clog
						<< "n=" << i
						<< ",bs=" << block_size
						<< ",nblocks=" << nblocks
						<< std::endl;
						*/
					for (int k=0; k<nblocks; ++k) {
						for (int j=0; j<stride; ++j) {
							const int offset = block_size*k + j;
							/*
							#ifndef NDEBUG
							std::clog << "FFT: extent=" << extent
								<< ",stride=" << stride
								<< ",offset=" << offset
								<< ",check=" << (offset + stride*(extent-1))
								<< ",dir=" << int(dir)
								<< std::endl;
							#endif
							*/
							this->_transforms[i].transform(
								rhs.data()+offset,
								stride,
								dir == apmath::Fourier_direction::Forward
								? gsl_fft_forward
								: gsl_fft_backward,
								workspace[i]
							);
						}
					}
				}
				return rhs;
			}

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

#endif // APMATH_FOURIER_GSL_HH

