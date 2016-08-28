#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_fft_complex_float.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_real_float.h>
#include <blitz/array.h>

namespace autoreg {

	enum struct Fourier_domain { Real, Complex };

	template <class T, Fourier_domain D>
	struct Fourier_workspace;

	template <>
	struct Fourier_workspace<double, Fourier_domain::Complex> {

		typedef gsl_fft_complex_workspace workspace_type;

		explicit Fourier_workspace(size_t n)
		    : _workspace(gsl_fft_complex_workspace_alloc(n)) {}
		~Fourier_workspace() { gsl_fft_complex_workspace_free(_workspace); }

		operator const workspace_type*() const { return _workspace; }
		operator workspace_type*() { return _workspace; }
		size_t
		size() const {
			return _workspace->n;
		}

	private:
		workspace_type* _workspace;
	};

	template <>
	struct Fourier_workspace<float, Fourier_domain::Complex> {

		typedef gsl_fft_complex_workspace_float workspace_type;

		explicit Fourier_workspace(size_t n)
		    : _workspace(gsl_fft_complex_workspace_float_alloc(n)) {}
		~Fourier_workspace() {
			gsl_fft_complex_workspace_float_free(_workspace);
		}

		operator const workspace_type*() const { return _workspace; }
		operator workspace_type*() { return _workspace; }
		size_t
		size() const {
			return _workspace->n;
		}

	private:
		workspace_type* _workspace;
	};

	template <>
	struct Fourier_workspace<double, Fourier_domain::Real> {

		typedef gsl_fft_real_workspace workspace_type;

		explicit Fourier_workspace(size_t n)
		    : _workspace(gsl_fft_real_workspace_alloc(n)) {}
		~Fourier_workspace() { gsl_fft_real_workspace_free(_workspace); }

		operator const workspace_type*() const { return _workspace; }
		operator workspace_type*() { return _workspace; }
		size_t
		size() const {
			return _workspace->n;
		}

	private:
		workspace_type* _workspace;
	};

	template <>
	struct Fourier_workspace<float, Fourier_domain::Real> {

		typedef gsl_fft_real_workspace_float workspace_type;

		explicit Fourier_workspace(size_t n)
		    : _workspace(gsl_fft_real_workspace_float_alloc(n)) {}
		~Fourier_workspace() { gsl_fft_real_workspace_float_free(_workspace); }

		operator const workspace_type*() const { return _workspace; }
		operator workspace_type*() { return _workspace; }
		size_t
		size() const {
			return _workspace->n;
		}

	private:
		workspace_type* _workspace;
	};

	template <class T, Fourier_domain D>
	struct Fourier_wavetable;

	template <>
	struct Fourier_wavetable<double, Fourier_domain::Complex> {

		typedef gsl_fft_complex_wavetable wavetable_type;

		explicit Fourier_wavetable(size_t n)
		    : _wavetable(gsl_fft_complex_wavetable_alloc(n)) {}
		~Fourier_wavetable() { gsl_fft_complex_wavetable_free(_wavetable); }

		operator const wavetable_type*() const { return _wavetable; }
		operator wavetable_type*() { return _wavetable; }
		size_t
		size() const {
			return _wavetable->n;
		}

	private:
		wavetable_type* _wavetable;
	};

	template <>
	struct Fourier_wavetable<float, Fourier_domain::Complex> {

		typedef gsl_fft_complex_wavetable_float wavetable_type;

		explicit Fourier_wavetable(size_t n)
		    : _wavetable(gsl_fft_complex_wavetable_float_alloc(n)) {}
		~Fourier_wavetable() {
			gsl_fft_complex_wavetable_float_free(_wavetable);
		}

		operator const wavetable_type*() const { return _wavetable; }
		operator wavetable_type*() { return _wavetable; }
		size_t
		size() const {
			return _wavetable->n;
		}

	private:
		wavetable_type* _wavetable;
	};

	template <>
	struct Fourier_wavetable<double, Fourier_domain::Real> {

		typedef gsl_fft_real_wavetable wavetable_type;

		explicit Fourier_wavetable(size_t n)
		    : _wavetable(gsl_fft_real_wavetable_alloc(n)) {}
		~Fourier_wavetable() { gsl_fft_real_wavetable_free(_wavetable); }

		operator const wavetable_type*() const { return _wavetable; }
		operator wavetable_type*() { return _wavetable; }
		size_t
		size() const {
			return _wavetable->n;
		}

	private:
		wavetable_type* _wavetable;
	};

	template <>
	struct Fourier_wavetable<float, Fourier_domain::Real> {

		typedef gsl_fft_real_wavetable_float wavetable_type;

		explicit Fourier_wavetable(size_t n)
		    : _wavetable(gsl_fft_real_wavetable_float_alloc(n)) {}
		~Fourier_wavetable() { gsl_fft_real_wavetable_float_free(_wavetable); }

		operator const wavetable_type*() const { return _wavetable; }
		operator wavetable_type*() { return _wavetable; }
		size_t
		size() const {
			return _wavetable->n;
		}

	private:
		wavetable_type* _wavetable;
	};

	template <class T, Fourier_domain D>
	struct Basic_fourier_transform;

	template <>
	struct Basic_fourier_transform<double, Fourier_domain::Complex> {

		typedef Fourier_wavetable<double, Fourier_domain::Complex>
		    wavetable_type;
		typedef Fourier_workspace<double, Fourier_domain::Complex>
		    workspace_type;

		explicit Basic_fourier_transform(size_t n)
		    : _wavetable(n), _workspace(n) {}

		template <class T>
		void
		forward(T* rhs, size_t stride) {
			gsl_fft_complex_transform(rhs, stride, _wavetable.size(),
			                          _wavetable, _workspace);
		}

		template <class T>
		void
		backward(T* rhs, size_t stride) {
			gsl_fft_complex_transform_backward(rhs, stride, _wavetable.size(),
			                                   _wavetable, _workspace);
		}

	private:
		wavetable_type _wavetable;
		workspace_type _workspace;
	};

	template <>
	struct Basic_fourier_transform<float, Fourier_domain::Complex> {

		typedef Fourier_wavetable<float, Fourier_domain::Complex>
		    wavetable_type;
		typedef Fourier_workspace<float, Fourier_domain::Complex>
		    workspace_type;

		explicit Basic_fourier_transform(size_t n)
		    : _wavetable(n), _workspace(n) {}

		template <class T>
		void
		forward(T* rhs, size_t stride) {
			gsl_fft_complex_float_transform(rhs, stride, _wavetable.size(),
			                                _wavetable, _workspace);
		}

		template <class T>
		void
		backward(T* rhs, size_t stride) {
			gsl_fft_complex_float_transform_backward(
			    rhs, stride, _wavetable.size(), _wavetable, _workspace);
		}

	private:
		wavetable_type _wavetable;
		workspace_type _workspace;
	};

	template <class T, int N, Fourier_domain D>
	struct Fourier_transform {

		explicit Fourier_transform(blitz::TinyVector<int, N> shape) {
			for (int i = 0; i < N; ++i) _transforms.emplace_back(shape(i));
		}

		template <class X>
		blitz::Array<X, N>
		forward(blitz::Array<X, N> rhs) {
			blitz::Array<X, N> result(rhs.copy());
			for (int i = 0; i < N; ++i) {
				_transforms[i].forward(result.data(), result.stride(i));
			}
			return result;
		}

	private:
		std::vector<Basic_fourier_transform<T, D>> _transforms;
	};
}
