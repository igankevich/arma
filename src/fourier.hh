#include <gsl/gsl_fft.h>
#include <blitz/array.h>

namespace autoreg {

	enum Domain {
		Real,
		Complex
	};

	template<class T, int N>
	struct Fourier_transform {
	private:
		gsl_fft_complex_wavetable* _wavetable;
		gsl_fft_complex_workspace* _workspace;
	};

}
