#ifndef PROFILE_HH
#define PROFILE_HH

#if ARMA_PROFILE
#include <chrono>
#include <iostream>
#include <sstream>

namespace arma {

	template <class Func>
	inline void
	__profile(const char* name, Func func) {
		using namespace std::chrono;
		auto t0 = high_resolution_clock::now();
		func();
		auto t1 = high_resolution_clock::now();
		auto ns = duration_cast<nanoseconds>(t1 - t0);
		std::stringstream msg;
		msg << name << ' ' << ns.count() << "ns\n";
		std::clog << msg.rdbuf();
	}

}

#define ARMA_PROFILE_FUNC(func) \
	::arma::__profile(#func, [&](){ func; })
#define ARMA_PROFILE_BLOCK(name, block) \
	::arma::__profile(name, [&](){ block; })

#else
namespace arma {

	template <class Func>
	inline void
	__profile(const char*, Func func) {
		func();
	}

}
#define ARMA_PROFILE_FUNC(func) func;
#define ARMA_PROFILE_BLOCK(name, block) block
#endif

#endif // PROFILE_HH
