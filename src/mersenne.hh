#ifndef MERSENNE_HH
#define MERSENNE_HH

#include <sstream>
#include <stdexcept>

/// @file
/// Implementation of Mersenne Twister pseudo-random number generator.
/// The only advantage of this algorithm is that it can generate
/// uncorreltaed sequences in parallel. In case of serial programme
/// any PRNG with long period can be used.

namespace autoreg {

	typedef struct{
	  unsigned int matrix_a;
	  unsigned int mask_b;
	  unsigned int mask_c;
	  unsigned int seed;
	} mt_struct_stripped;

	const unsigned int DCMT_SEED    = 4172;
	const size_t       MT_RNG_COUNT = 4096;
	const int          MT_MM        = 9;
	const int          MT_NN        = 19;
	const unsigned int MT_WMASK     = 0xFFFFFFFFU;
	const unsigned int MT_UMASK     = 0xFFFFFFFEU;
	const unsigned int MT_LMASK     = 0x1U;
	const int          MT_SHIFT0    = 12;
	const int          MT_SHIFTB    = 7;
	const int          MT_SHIFTC    = 15;
	const int          MT_SHIFT1    = 18;

	/// Read Mersenne Twister parameters from file @filename into @h_MT.
	void read_mt_params(mt_struct_stripped* h_MT, const char *filename) {
	
	  	std::ifstream in(filename, std::ios::binary);
		if (!in.is_open()) {
			std::stringstream msg;
			msg << "File not found: " << filename;
			throw std::runtime_error(msg.str());
		}
		
		for (size_t i = 0; i < MT_RNG_COUNT; i++) {
		    in.read((char*)&h_MT[i], sizeof(mt_struct_stripped));
		}
	
	#ifdef DISABLE_RANDOM_SEED
		const unsigned int seed = 0u;
	#else
		const unsigned int seed = time(0);
	#endif
	
		for (size_t i = 0; i < MT_RNG_COUNT; i++) {
		    h_MT[i].seed = seed;
		}
	}

}

#endif // MERSENNE_HH
