#ifndef PROFILE_COUNTERS_HH
#define PROFILE_COUNTERS_HH

#include "profile.hh"

#define CNT_WINDOWFUNC 0
#define CNT_SECONDFUNC 1
#define CNT_FFT 2
#define CNT_DEVTOHOST_COPY 3

namespace arma {

	inline void
	register_all_counters() {
		arma::register_counter(CNT_WINDOWFUNC, "window_function");
		arma::register_counter(CNT_SECONDFUNC, "second_function");
		arma::register_counter(CNT_FFT, "fft");
		arma::register_counter(CNT_DEVTOHOST_COPY, "dev_to_host_copy");
	}

}

#endif // PROFILE_COUNTERS_HH
