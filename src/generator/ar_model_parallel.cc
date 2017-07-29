#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <fstream>
#include <iterator>
#include <mutex>
#if ARMA_OPENMP
#include <omp.h>
#endif
#include <stdexcept>
#include <vector>
#include "parallel_mt.hh"
#include "config.hh"
#include "errors.hh"
#include "util.hh"

using namespace arma;

namespace {

	struct Partition {

		Partition() = default;

		Partition(
			Shape3D ijk_,
			const blitz::RectDomain<3>& r,
			const prng::mt_config& conf,
			uint32_t seed
		):
		ijk(ijk_), rect(r), prng(conf)
		{ prng.seed(seed); }

		friend std::ostream&
		operator<<(std::ostream& out, const Partition& rhs) {
			return out << rhs.ijk << ": " << rhs.rect;
		}

		Shape3D
		shape() const {
			return get_shape(rect);
		}

		Shape3D ijk;
		blitz::RectDomain<3> rect;
		prng::parallel_mt prng;
	};

	template<class Result>
	void
	read_parallel_mt_config(const char* filename, Result result) {
		std::ifstream in(filename);
		if (!in.is_open()) {
			throw std::runtime_error("bad file");
		}
		std::copy(
			std::istream_iterator<prng::mt_config>(in),
			std::istream_iterator<prng::mt_config>(),
			result
		);
	}

	std::vector<Partition>
	partition(
		Shape3D nparts,
		Shape3D partshape,
		Shape3D shape,
		const std::vector<prng::mt_config>& prng_config,
		const uint32_t seed
	) {
		std::vector<Partition> parts;
		const int nt = nparts(0);
		const int nx = nparts(1);
		const int ny = nparts(2);
		for (int i=0; i<nt; ++i) {
			for (int j=0; j<nx; ++j) {
				for (int k=0; k<ny; ++k) {
					const Shape3D ijk(i, j, k);
					const Shape3D lower = blitz::min(ijk * partshape, shape);
					const Shape3D upper = blitz::min((ijk+1) * partshape, shape) - 1;
					parts.emplace_back(
						ijk,
						blitz::RectDomain<3>(lower, upper),
						prng_config[parts.size()],
						seed
					);
				}
			}
		}
		return parts;
	}

	arma::Shape3D
	get_partition_shape(
		Shape3D partition_shape,
		Shape3D grid_shape,
		Shape3D order,
		int nprngs
	) {
		using blitz::product;
		using std::min;
		using std::cbrt;
		Shape3D ret;
		if (product(partition_shape) > 0) {
			ret = partition_shape;
		} else {
			const Shape3D guess1 = blitz::max(
				order * 2,
				Shape3D(10, 10, 10)
			);
			#if ARMA_OPENMP
			const int parallelism = min(omp_get_max_threads(), nprngs);
			#else
			const int parallelism = nprngs;
			#endif
			const int npar = std::max(1, 7*int(cbrt(parallelism)));
			const Shape3D guess2 = blitz::div_ceil(
				grid_shape,
				Shape3D(npar, npar, npar)
			);
			ret = blitz::min(guess1, guess2) + blitz::abs(guess1 - guess2) / 2;
		}
		return ret;
	}

}

template <class T>
arma::Array3D<T>
arma::generator::AR_model<T>::do_generate() {
	const T var_wn = this->white_noise_variance();
	write_key_value(std::clog, "White noise variance", var_wn);
	using blitz::RectDomain;
	using blitz::product;
	/// 1. Read parallel Mersenne Twister states.
	std::vector<prng::mt_config> prng_config;
	read_parallel_mt_config(
		MT_CONFIG_FILE,
		std::back_inserter(prng_config)
	);
	const int nprngs = prng_config.size();
	if (nprngs == 0) {
		throw PRNG_error("bad number of MT configs", nprngs, 0);
	}
	/// 2. Partition the data.
	const Shape3D shape = this->_outgrid.size();
	const Shape3D partshape = get_partition_shape(
		this->_partition,
		this->grid().num_points(),
		this->order(),
		nprngs
	);
	const Shape3D nparts = blitz::div_ceil(shape, partshape);
	const int ntotal = product(nparts);
	if (prng_config.size() < size_t(product(nparts))) {
		throw PRNG_error("bad number of MT configs", nprngs, ntotal);
	}
	write_key_value(std::clog, "Partition size", partshape);
	std::vector<Partition> parts = partition(
		nparts,
		partshape,
		shape,
		prng_config,
		this->newseed()
	);
	Array3D<bool> completed(nparts);
	Array3D<T> zeta(shape), eps(shape);
	std::condition_variable cv;
	std::mutex mtx;
	std::atomic<int> nfinished(0);
	/// 3. Put all partitions in a queue and process them in parallel.
	/// Each thread traverses the queue looking for partitions depedent
	/// partitions of which has been computed. When eligible partition is
	/// found, it is removed from the queue and computed and its status is
	/// updated in a separate map.
	#pragma omp parallel
	{
		std::unique_lock<std::mutex> lock(mtx);
		while (!parts.empty()) {
			typename std::vector<Partition>::iterator result;
			cv.wait(lock, [&result,&parts,&completed] () {
				result = std::find_if(
					parts.begin(),
					parts.end(),
					[&completed] (const Partition& part) {
						completed(part.ijk) = true;
						Shape3D ijk0 = blitz::max(0, part.ijk - 1);
						bool all_completed = blitz::all(
							completed(blitz::RectDomain<3>(ijk0, part.ijk))
						);
						completed(part.ijk) = false;
						return all_completed;
					}
				);
				return result != parts.end() || parts.empty();
			});
			if (parts.empty()) {
				break;
			}
			Partition part = *result;
			parts.erase(result);
			lock.unlock();
			eps(part.rect) = generate_white_noise(
				part.shape(),
				var_wn,
				std::ref(part.prng)
			);
			this->generate_surface(zeta, eps, part.rect);
			lock.lock();
			std::clog
				<< "\rFinished part ["
				<< ++nfinished << '/' << ntotal << ']';
			completed(part.ijk) = true;
			cv.notify_all();
		}
	}
	std::clog << std::endl;
	return zeta;
}