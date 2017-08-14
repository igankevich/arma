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
#include <thread>
#include <string>
#include <stdexcept>
#include <vector>
#include "parallel_mt.hh"
#include "config.hh"
#include "errors.hh"
#include "util.hh"
#include "io/binary_stream.hh"
#include "profile_counters.hh"

using namespace arma;

namespace {

	struct Partition {

		Partition() = default;

		Partition(Shape3D ijk_, const blitz::RectDomain<3>& r):
		ijk(ijk_),
		rect(r)
		{}

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
	};

	std::vector<Partition>
	partition(
		Shape3D nparts,
		Shape3D partshape,
		Shape3D shape
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
					parts.emplace_back(ijk, blitz::RectDomain<3>(lower, upper));
				}
			}
		}
		return parts;
	}

	template <class T, class Generator>
	Array3D<T>
	generate_white_noise(const Shape3D& size, T variance, Generator generator) {
		std::normal_distribution<T> normal(T(0), std::sqrt(variance));
		Array3D<T> eps(size);
		std::generate_n(
			eps.data(),
			eps.numElements(),
			std::bind(normal, generator)
		);
		return eps;
	}

}

template <class T>
arma::Array3D<T>
arma::generator::AR_model<T>::do_generate() {
	const T var_wn = this->white_noise_variance();
	write_key_value(std::clog, "White noise variance", var_wn);
	if (var_wn < T(0)) {
		throw std::invalid_argument("variance is less than zero");
	}
	using blitz::RectDomain;
	using blitz::product;
	using std::min;
	/// 1. Read parallel Mersenne Twister states.
	const size_t nthreads = std::max(1, omp_get_max_threads());
	std::vector<prng::parallel_mt> mts =
		prng::read_parallel_mts(MT_CONFIG_FILE, nthreads, this->_noseed);
	/// 2. Partition the data.
	const Shape3D shape = this->_outgrid.size();
	const Shape3D partshape = get_partition_shape(
		this->_partition,
		this->grid().num_points(),
		this->order(),
		nthreads
	);
	const Shape3D nparts = blitz::div_ceil(shape, partshape);
	const int ntotal = product(nparts);
	write_key_value(std::clog, "Partition size", partshape);
	std::vector<Partition> parts = partition(nparts, partshape, shape);
	Array3D<bool> completed(nparts);
	Array3D<T> zeta(shape);
	std::condition_variable cv;
	std::mutex mtx;
	std::atomic<int> nfinished(0);
	const bool writing_in_parallel = this->writes_in_parallel();
	// how many parts are computed along t dimension
	Array1D<int> parts_per_slice_completed(nparts(0));
	parts_per_slice_completed = 0;
	std::thread writer;
	if (writing_in_parallel) {
		writer = std::thread([&] () {
			using blitz::Range;
			std::string filename = get_surface_filename(Output_flags::Binary);
			io::Binary_stream out(filename);
			int part_t = 0;
			const int nparts_t = nparts(0);
			const int nparts_per_slice = nparts(1)*nparts(2);
			std::unique_lock<std::mutex> lock(mtx);
			while (part_t < nparts_t) {
				cv.wait(lock, [&] () {
					return parts_per_slice_completed(part_t) == nparts_per_slice;
				});
				lock.unlock();
				const int t0 = part_t*partshape(0);
				const int t1 = std::min(t0 + partshape(0), shape(0)) - 1;
				print_progress("wrote slice", t1+1, shape(0));
				ARMA_EVENT_START("write_surface", "io", 0);
				out.write(Array3D<T>(
					zeta,
					Range(t0,t1),
					Range::all(),
					Range::all()
				));
				ARMA_EVENT_END("write_surface", "io", 0);
				++part_t;
				lock.lock();
			}
			ARMA_EVENT_START("write_surface", "io", 0);
			out.close();
			ARMA_EVENT_END("write_surface", "io", 0);
		});
	}
	/// 3. Put all partitions in a queue and process them in parallel.
	/// Each thread traverses the queue looking for partitions depedent
	/// partitions of which has been computed. When eligible partition is
	/// found, it is removed from the queue and computed and its status is
	/// updated in a separate map.
	#pragma omp parallel
	{
		const int thread_no = omp_get_thread_num();
		prng::parallel_mt& mt = mts[thread_no];
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
			ARMA_EVENT_START("generate_surface", "omp", thread_no);
			zeta(part.rect) = generate_white_noise(
				part.shape(),
				var_wn,
				std::ref(mt)
			);
			this->generate_surface(zeta, part.rect);
			ARMA_EVENT_END("generate_surface", "omp", thread_no);
			lock.lock();
			print_progress("generated part", ++nfinished, ntotal);
			completed(part.ijk) = true;
			if (writing_in_parallel) {
				++parts_per_slice_completed(part.ijk(0));
			}
			cv.notify_all();
		}
	}
	if (writer.joinable()) {
		writer.join();
	}
	return zeta;
}
