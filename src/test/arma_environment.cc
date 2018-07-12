#include "arma_environment.hh"

#if ARMA_OPENCL
#include <opencl/opencl.hh>
#endif

#include "config.hh"

#if defined(ARMA_CLFFT)
#include <clFFT.h>
#endif

void
ARMA_environment::SetUp() {
	#if ARMA_OPENCL
	::arma::opencl::init();
	#endif
}

void
ARMA_environment::TearDown() {
	#if defined(ARMA_CLFFT)
	#define CHECK(x) ::cl::detail::errHandler((x), #x);
	CHECK(clfftTeardown());
	#undef CHECK
	#endif
}
