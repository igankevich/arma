#ifndef BITS_FOURIER_GSL_HH
#define BITS_FOURIER_GSL_HH

#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_complex_float.h>
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
					!std::is_same<std::complex<elem_type>, T>::value,
					"bad types"
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


}

#endif // BITS_FOURIER_GSL_HH
