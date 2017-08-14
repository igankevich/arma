#ifndef PROFILE_HH
#define PROFILE_HH

#if ARMA_PROFILE
#include <chrono>
#include <iostream>
#include <sstream>
#include <ostream>

namespace arma {

	typedef std::chrono::high_resolution_clock::rep counter_type;

	extern counter_type __counters[4096 / sizeof(counter_type)];

	template <class Func>
	inline void
	__profile(const char* name, Func func) {
		using namespace std::chrono;
		auto t0 = high_resolution_clock::now();
		func();
		auto t1 = high_resolution_clock::now();
		auto us = duration_cast<microseconds>(t1 - t0);
		std::stringstream msg;
		msg << "prfl" << ' ' << name << ' ' << us.count() << "us\n";
		std::clog << msg.rdbuf();
	}

	template <class Func>
	inline void
	__profile_cnt(counter_type cnt_name, Func func) {
		using namespace std::chrono;
		auto t0 = high_resolution_clock::now();
		func();
		auto t1 = high_resolution_clock::now();
		auto us = duration_cast<microseconds>(t1 - t0);
		__counters[cnt_name] += us.count();
	}

	void
	print_counters(std::ostream& out);

	void
	register_counter(size_t idx, std::string name);

	namespace profile {

		extern const std::chrono::high_resolution_clock::time_point programme_start;

		void
		thread_event(
			const char* name,
			const char* state,
			const char* thread_name,
			const int thread_no
		);

	}

}

#define ARMA_PROFILE_FUNC(func) \
	::arma::__profile(#func, [&](){ func; })
#define ARMA_PROFILE_BLOCK(name, block) \
	::arma::__profile(name, [&](){ block; })
#define ARMA_PROFILE_CNT(cnt, block) \
	::arma::__profile_cnt(cnt, [&](){ block; })
#define ARMA_PROFILE_START(name) \
	const auto name##_t0 = ::std::chrono::high_resolution_clock::now()
#define ARMA_PROFILE_END(name) \
	{ \
		auto name##_t1 = ::std::chrono::high_resolution_clock::now(); \
		auto us = ::std::chrono::duration_cast<::std::chrono::microseconds> \
			(name##_t1 - name##_t0); \
		::std::stringstream msg; \
		msg << "prfl" << ' ' << #name << ' ' << us.count() << "us\n"; \
		::std::clog << msg.rdbuf(); \
	}
#define ARMA_EVENT_START(name, thread_name, thread_no) \
	::arma::profile::thread_event(name, "strt", thread_name, thread_no)
#define ARMA_EVENT_END(name, thread_name, thread_no) \
	::arma::profile::thread_event(name, "end", thread_name, thread_no)

#else
#define ARMA_PROFILE_FUNC(func) func;
#define ARMA_PROFILE_BLOCK(name, block) block
#define ARMA_PROFILE_CNT(cnt, block) block
#define ARMA_PROFILE_START(name)
#define ARMA_PROFILE_END(name)
#define ARMA_EVENT_START(name)
#define ARMA_EVENT_END(name)
#endif

#endif // PROFILE_HH
