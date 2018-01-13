#ifndef REGISTER_ALL_HH
#define REGISTER_ALL_HH

#if defined(ARMA_CLFFT)
#include <clFFT.h>
#endif

#include "config.hh"

#include "generator/ar_model.hh"
#include "generator/ma_model.hh"
#include "generator/arma_model.hh"
#include "generator/plain_wave_model.hh"
#include "generator/lh_model.hh"

#include "velocity/high_amplitude_solver.hh"
#include "velocity/linear_solver.hh"
#include "velocity/plain_wave_solver.hh"
#if defined(WITH_SMALL_AMPLITUDE_SOLVER)
#include "velocity/small_amplitude_solver.hh"
#endif

#if ARMA_OPENCL
#include "opencl/opencl.hh"
#endif

#include "profile.hh"
#if ARMA_PROFILE
#include "profile_counters.hh"
#endif

void
register_all_counters() {
	#if ARMA_PROFILE
	using namespace arma;
	register_counter(CNT_HARTS_G1, "harts_g1");
	register_counter(CNT_HARTS_G2, "harts_g2");
	register_counter(CNT_HARTS_FFT, "harts_fft");
	register_counter(CNT_HARTS_COPY_TO_HOST, "harts_copy_to_host");
	register_counter(CNT_COPY_TO_HOST, "copy_to_host");
	register_counter(CNT_WRITE_SURFACE, "write_surface");
	register_counter(CNT_BSC_COPY, "bsc_copy");
	register_counter(CNT_BSC_MARSHALLING, "bsc_marshalling");
	#endif
}

template <class T>
void
register_all_solvers(arma::ARMA_driver<T>& drv) {
	using namespace ::arma::velocity;
	drv.template register_solver<Linear_solver<T>>("linear");
	drv.template register_solver<Plain_wave_solver<T>>("plain");
	drv.template register_solver<High_amplitude_solver<T>>("high_amplitude");
	#if defined(WITH_SMALL_AMPLITUDE_SOLVER)
	drv.template register_solver<Small_amplitude_solver<T>>("small_amplitude");
	#endif
}

template <class T>
void
register_all_models(arma::ARMA_driver<T>& drv) {
	using namespace ::arma::generator;
	drv.template register_model<AR_model<T>>("AR");
	drv.template register_model<MA_model<T>>("MA");
	drv.template register_model<ARMA_model<T>>("ARMA");
	drv.template register_model<Plain_wave_model<T>>("plain_wave");
	drv.template register_model<Longuet_Higgins_model<T>>("LH");
}

#endif // vim:filetype=cpp
