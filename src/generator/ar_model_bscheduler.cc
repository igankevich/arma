#include <thread>
#include <vector>

#include <bscheduler/api.hh>

#include "config.hh"
#include "parallel_mt.hh"
#include "profile.hh"

namespace {

template <class T>
class ar_partition_kernel: public bsc::kernel {

private:
	typedef arma::Shape3D shape_type;
	typedef arma::Array3D<T> array_type;
	typedef blitz::RectDomain<3> rect_type;
	typedef arma::prng::parallel_mt generator_type;
	typedef std::vector<generator_type> mts_type;

private:
	/// Partition start.
	shape_type _lower;
	/// Partition end.
	shape_type _upper;
	/// Shape of the wavy surface.
	shape_type _shape;
	/// Wavy surface.
	array_type& _zeta;
	/// Parallel Mersenne Twisters.
	mts_type& _mts;
	/// MT index.
	int _index = 0;
	/// White noise variance.
	T _varwn = 0;
	/// Number of dependent kernels.
	int _nsubordinates = 0;

public:

	ar_partition_kernel() = default;

	inline
	ar_partition_kernel(
		shape_type lower,
		shape_type upper,
		shape_type shape,
		T varwn,
		int index,
		array_type& zeta,
		mts_type& mts
	):
	_lower(blitz::min(lower, shape)),
	_upper(blitz::min(upper, shape)),
	_shape(shape),
	_zeta(zeta),
	_mts(mts),
	_index(index),
	_varwn(varwn)
	{}

	void
	act() override {
		using blitz::shape;
		rect_type rect(this->_lower, this->_upper);
		shape_type partshape = this->get_part_shape();
		this->_zeta(rect) = generate_white_noise(
			partshape,
			this->_varwn,
			std::ref(this->mersenne_twister())
		);
		generate_surface(this->_zeta, rect);
		/// Launch all dependent kernels.
		int idx = this->_index;
		std::vector<ar_partition_kernel*> kernels;
		send_subordinate(shape(partshape(0), 0, 0), idx, kernels);
		send_subordinate(shape(0, partshape(1), 0), idx, kernels);
		send_subordinate(shape(0, 0, partshape(2)), idx, kernels);
		send_subordinate(shape(partshape(0), partshape(1), 0), idx, kernels);
		send_subordinate(shape(0, partshape(1), partshape(2)), idx, kernels);
		send_subordinate(shape(partshape(0), 0, partshape(2)), idx, kernels);
		send_subordinate(
			shape(partshape(0), partshape(1), partshape(2)),
			idx,
			kernels
		);
		this->_nsubordinates = kernels.size();
		for (ar_partition_kernel* k : kernels) {
			bsc::upstream(this, k);
		}
	}

private:

	inline void
	send_subordinate(
		shape_type shift,
		int& index,
		std::vector<ar_partition_kernel*>& kernels
	) {
		using blitz::min;
		using blitz::product;
		shape_type partshape = this->get_part_shape();
		shape_type new_lower = min(this->_lower + shift, this->_shape);
		shape_type new_upper = min(this->_upper + shift, this->_shape);
		if (product(new_upper - new_lower) == 0) {
			// empty part
			return;
		}
		kernels.push_back(new ar_partition_kernel(
			new_lower,
			new_upper,
			this->_shape,
			this->_varwn,
			++index,
			this->_zeta,
			this->_mts
		));
	}

	inline shape_type
	get_part_shape() const noexcept {
		return this->_upper - this->_lower + 1;
	}

	inline generator_type&
	mersenne_twister() noexcept {
		return this->_mts[this->_index];
	}

};

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
	/// 1. Partition the data.
	const size_t nthreads = std::max(1u, std::thread::hardware_concurrency());
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
	/// 2. Read parallel Mersenne Twister state for each kernel.
	std::vector<prng::parallel_mt> mts =
		prng::read_parallel_mts(MT_CONFIG_FILE, ntotal, this->_noseed);
	Array3D<T> zeta(shape);
	return zeta;
}


