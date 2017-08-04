#ifndef WHITE_NOISE_HH
#define WHITE_NOISE_HH

#include <cmath>
#if ARMA_OPENMP || ARMA_OPENCL
#include <omp.h>
#endif

#include "types.hh"
#include "parallel_mt.hh"
#include "config.hh"

namespace arma {

	namespace prng {

		/**
		\brief Generate white noise via Mersenne Twister algorithm.

		Convert to normal distribution via Box---Muller transform.
		Uses parallel MT implementation if OpenMP is enabled.
		*/
		template <class T, int N, class Dist>
		blitz::Array<T,N>
		generate_white_noise(
			const blitz::TinyVector<int,N>& shape,
			bool noseed,
			Dist dist
		) {
			/// 1. Read parallel Mersenne Twister states.
			#if ARMA_OPENMP || ARMA_OPENCL
			const size_t nthreads = std::max(1, omp_get_max_threads());
			#else
			const size_t nthreads = 1;
			#endif
			std::vector<parallel_mt> mts =
				read_parallel_mts(MT_CONFIG_FILE, nthreads, noseed);
			/// 2. Generate white noise in parallel.
			blitz::Array<T,N> eps(shape);
			const int n = eps.numElements();
			#if ARMA_OPENMP || ARMA_OPENCL
			#pragma omp parallel
			#endif
			{
				#if ARMA_OPENMP || ARMA_OPENCL
				prng::parallel_mt& mt = mts[omp_get_thread_num()];
				#else
				prng::parallel_mt& mt = mts[0];
				#endif
				#if ARMA_OPENMP || ARMA_OPENCL
				#pragma omp for
				#endif
				for (int i=0; i<n; ++i) {
					eps.data()[i] = dist(mt);
				}
			}
			return eps;
		}


	}

}

#endif // WHITE_NOISE_HH
