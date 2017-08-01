#ifndef FOURIER_HH
#define FOURIER_HH

/* disable opencl until it works
#if ARMA_OPENCL
#include "apmath/fourier_opencl.hh"
#else
#include "apmath/fourier_gsl.hh"
#endif
*/

#if ARMA_OPENCL
#include "opencl/opencl.hh"
#endif
#include "apmath/fourier_gsl.hh"

#endif // FOURIER_HH
