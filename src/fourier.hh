#ifndef FOURIER_HH
#define FOURIER_HH

#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_complex_float.h>
#include <blitz/array.h>
#include <complex>
#include <vector>
#include <type_traits>

namespace arma {

	namespace bits {

		template<
			class Workspace,
			Workspace* (*Alloc)(size_t),
			void (*Free)(Workspace*)
		>
		class Fourier_building_block {

			typedef Workspace workspace_type;
			workspace_type* _workspace;

		public:

			typedef Workspace gsl_type;

			explicit
			Fourier_building_block(size_t n):
			_workspace(Alloc(n))
			{}

			Fourier_building_block() = delete;

			Fourier_building_block(const Fourier_building_block& rhs) = delete;

			Fourier_building_block&
			operator=(const Fourier_building_block& rhs) = delete;

			Fourier_building_block(Fourier_building_block&& rhs):
			_workspace(rhs._workspace)
			{ rhs._workspace = nullptr; }

			~Fourier_building_block() {
				Free(_workspace);
				_workspace = nullptr;
			}

			operator const workspace_type*() const { return _workspace; }
			operator workspace_type*() { return _workspace; }

			size_t
			size() const noexcept {
				return _workspace ? _workspace->n : 0;
			}

		};


		template<
			class Wavetable,
			class Workspace,
			class Array,
			class Value,
			int (*Transform)(
				Array,
				const size_t,
				const size_t,
				const typename Wavetable::gsl_type*,
				typename Workspace::gsl_type*,
				const gsl_fft_direction
			)
		>
		class Basic_fourier_transform {

			typedef Wavetable wavetable_type;
			typedef Workspace workspace_type;
			typedef Array array_type;
			typedef Value value_type;

			wavetable_type _wavetable;

		public:
			explicit
			Basic_fourier_transform(size_t n):
			_wavetable(n) {}

			Basic_fourier_transform() = delete;

			Basic_fourier_transform(const Basic_fourier_transform&) = delete;

			Basic_fourier_transform&
			operator=(const Basic_fourier_transform&) = delete;

			Basic_fourier_transform(Basic_fourier_transform&& rhs):
			_wavetable(std::move(rhs._wavetable))
			{}

			~Basic_fourier_transform() = default;

			template <class T>
			void
			transform(
				T* rhs,
				size_t stride,
				const gsl_fft_direction dir,
				workspace_type& workspace
			) {
				typedef typename std::remove_pointer<array_type>::type elem_type;
				static_assert(
					(
						std::is_same<std::complex<elem_type>, T>::value
						and sizeof(T) == sizeof(value_type)
						and alignof(T) == alignof(value_type)
					)
					or
					!std::is_same<std::complex<elem_type>, T>::value
				);
				Transform(
					reinterpret_cast<array_type>(rhs),
					stride,
					this->_wavetable.size(),
					this->_wavetable,
					workspace,
					dir
				);
			}

			size_t
			size() const noexcept {
				return this->_wavetable.size();
			}

		};

		template <class T>
		struct Fourier_config {};

		template <>
		struct Fourier_config<std::complex<double>> {
			typedef Fourier_building_block<
				gsl_fft_complex_workspace,
				gsl_fft_complex_workspace_alloc,
				gsl_fft_complex_workspace_free
			> workspace_type;
			typedef Fourier_building_block<
				gsl_fft_complex_wavetable,
				gsl_fft_complex_wavetable_alloc,
				gsl_fft_complex_wavetable_free
			> wavetable_type;
			typedef Basic_fourier_transform<
				wavetable_type,
				workspace_type,
				gsl_complex_packed_array,
				gsl_complex,
				gsl_fft_complex_transform
			> transform_type;
		};

		template <>
		struct Fourier_config<std::complex<float>> {
			typedef Fourier_building_block<
				gsl_fft_complex_workspace_float,
				gsl_fft_complex_workspace_float_alloc,
				gsl_fft_complex_workspace_float_free
			> workspace_type;
			typedef Fourier_building_block<
				gsl_fft_complex_wavetable_float,
				gsl_fft_complex_wavetable_float_alloc,
				gsl_fft_complex_wavetable_float_free
			> wavetable_type;
			typedef Basic_fourier_transform<
				wavetable_type,
				workspace_type,
				gsl_complex_packed_array_float,
				gsl_complex_float,
				gsl_fft_complex_float_transform
			> transform_type;
		};

	}

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
		typedef blitz::Array<T,N> array_type;

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
			return transform(rhs, ws, gsl_fft_forward);
		}

		inline array_type
		forward(array_type rhs, workspace_type& workspace) {
			return transform(rhs, workspace, gsl_fft_forward);
		}

		inline array_type
		backward(array_type rhs) {
			workspace_type ws(this->new_workspace());
			return transform(rhs, ws, gsl_fft_backward);
		}

		inline array_type
		backward(array_type rhs, workspace_type& workspace) {
			return transform(rhs, workspace, gsl_fft_backward);
		}

		inline array_type
		transform(
			array_type rhs,
			workspace_type& workspace,
			gsl_fft_direction dir
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
							dir,
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

#endif // FOURIER_HH
