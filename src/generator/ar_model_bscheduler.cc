#include <thread>
#include <vector>

#include <bscheduler/api.hh>

#include "config.hh"
#include "parallel_mt.hh"
#include "profile.hh"
#include "util.hh"

#include "ar_model_wn.cc"

namespace {

	template <class T>
	class ar_partition_kernel: public bsc::kernel {

	private:
		typedef arma::Shape3D shape_type;
		typedef arma::Array3D<T> array_type;
		typedef blitz::RectDomain<3> rect_type;
		typedef arma::prng::parallel_mt generator_type;
		typedef arma::generator::AR_model<T> model_type;

	private:
		/// Partition start.
		shape_type _lower;
		/// Partition end.
		shape_type _upper;
		/// Part index in the counter array.
		shape_type _part;
		/// White noise variance.
		T _varwn = 0;
		/// AR mode reference.
		model_type& _model;
		/// Wavy surface.
		array_type& _zeta;
		/// Parallel Mersenne Twister.
		generator_type& _generator;

	public:

		ar_partition_kernel() = default;

		inline
		ar_partition_kernel(
			shape_type lower,
			shape_type upper,
			shape_type part,
			T varwn,
			array_type& zeta,
			generator_type& generator,
			model_type& model
		):
		_lower(lower),
		_upper(upper),
		_part(part),
		_varwn(varwn),
		_model(model),
		_zeta(zeta),
		_generator(generator)
		{}

		void
		act() override {
			using blitz::shape;
			rect_type rect(this->_lower, this->_upper);
			shape_type partshape = this->get_part_shape();
			this->_zeta(rect) = generate_white_noise(
				partshape,
				this->_varwn,
				std::ref(this->_generator)
			                    );
			this->_model.generate_surface(this->_zeta, rect);
			bsc::commit(this);
		}

		inline const shape_type&
		get_part_index() const noexcept {
			return this->_part;
		}

	private:

		inline shape_type
		get_part_shape() const noexcept {
			return this->_upper - this->_lower + 1;
		}

	};

	template <class T>
	class ar_master_kernel: public bsc::kernel {

	private:
		typedef arma::Array3D<unsigned char> counter3d_type;
		typedef arma::Shape3D shape_type;
		typedef arma::generator::AR_model<T> model_type;
		typedef ar_partition_kernel<T> slave_kernel;
		typedef blitz::RectDomain<3> rect_type;
		typedef arma::prng::parallel_mt generator_type;

	private:
		/// Three-dimensional array that stores the number of
		/// completed dependencies for each partition.
		/// Values range from 0 to 7.
		counter3d_type _counter;
		/// Shape of the partition.
		shape_type _partshape;
		/// The number of parts.
		shape_type _nparts;
		/// White noise variance.
		T _varwn = 0;
		/// AR model.
		model_type& _model;
		/// MT index.
		int _index = 0;
		int _nfinished = 0;
		int _ntotal = 0;

	public:
		ar_master_kernel(
			model_type& model,
			const shape_type& partshape,
			const shape_type& nparts
		):
		_counter(nparts),
		_partshape(partshape),
		_nparts(nparts),
		_model(model),
		_ntotal(blitz::product(nparts)) {
			this->init_counters();
			this->_varwn = this->_model.white_noise_variance();
			arma::write_key_value(std::clog, "White noise variance", this->_varwn);
			if (this->_varwn < T(0)) {
				throw std::invalid_argument("variance is less than zero");
			}
		}

		void
		act() override {
			// Launch the first kernel.
			send_subordinate(shape_type(0,0,0));
		}

		void
		react(bsc::kernel* child) override {
			using blitz::all;
			slave_kernel* slave = dynamic_cast<slave_kernel*>(child);
			const shape_type& lower = slave->get_part_index();
			arma::print_progress(
				"generated part",
				++this->_nfinished,
				this->_ntotal
			);
			if (all(lower+1 == this->_nparts)) {
				bsc::commit(this);
			} else {
				// send dependent kernel counters
				send_subordinate(shape_type(lower(0)+1, lower(1), lower(2)));
				send_subordinate(shape_type(lower(0), lower(1)+1, lower(2)));
				send_subordinate(shape_type(lower(0), lower(1), lower(2)+1));
				send_subordinate(shape_type(lower(0)+1, lower(1)+1, lower(2)));
				send_subordinate(shape_type(lower(0), lower(1)+1, lower(2)+1));
				send_subordinate(shape_type(lower(0)+1, lower(1), lower(2)+1));
				send_subordinate(shape_type(lower(0)+1, lower(1)+1, lower(2)+1));
			}
			/*
			const int nt = this->_counter.extent(0);
			for (int i=0; i<nt; ++i) {
				using blitz::Range;
				sys::log_message(
					"ar",
					"counter=_",
					this->_counter(i, Range::all(), Range::all())
				);
			}
			*/
		}

	private:

		inline void
		send_subordinate(const shape_type& part_index) {
			using blitz::any;
			if (any(part_index == this->_nparts)) {
				return;
			}
			if (++this->_counter(part_index) == 7) {
				//sys::log_message("ar", "part=_", part_index);
				// increment counter to prevent calculation
				// of the part multiple times
				++this->_counter(part_index);
				bsc::upstream(
					this,
					new slave_kernel(
						shape_type(0,0,0),
						this->partition_shape()-1,
						part_index,
						this->_varwn,
						this->_model._zeta,
						this->mersenne_twister(),
						this->_model
					)
				);
				++this->_index;
			}
		}

		void
		init_counters() {
			using blitz::Range;
			this->_counter = 0;
			// three dependencies for each face of the 3-d array
			this->_counter(0,Range::all(),Range::all()) = 4;
			this->_counter(Range::all(),0,Range::all()) = 4;
			this->_counter(Range::all(),Range::all(),0) = 4;
			// one dependency for each edge of the 3-d array
			this->_counter(0,0,Range::all()) = 6;
			this->_counter(0,Range::all(),0) = 6;
			this->_counter(Range::all(),0,0) = 6;
			// no dependencies for the first point
			// but we need to set it to 6 for an algorithm
			// to work
			this->_counter(0,0,0) = 6;
		}

		inline const shape_type&
		partition_shape() const noexcept {
			return this->_partshape;
		}

		inline const shape_type&
		zeta_shape() const noexcept {
			return this->_model._zeta.shape();
		}

		inline generator_type&
		mersenne_twister() noexcept {
			return this->_model._mts[this->_index];
		}

	};

}


template <class T>
void
arma::generator::AR_model<T>::act() {
	Basic_ARMA_model<T>::act();
	using blitz::RectDomain;
	using blitz::product;
	using std::min;
	/// 1. Partition the data.
	const size_t nthreads = std::max(1u, std::thread::hardware_concurrency());
	const Shape3D shape = this->_outgrid.size();
	const Shape3D partshape =
		get_partition_shape(
			this->_partition,
			this->grid().num_points(),
			this->order(),
			nthreads
		);
	const Shape3D nparts = blitz::div_ceil(shape, partshape);
	const int ntotal = product(nparts);
	write_key_value(std::clog, "Partition size", partshape);
	write_key_value(std::clog, "No. of parts", nparts);
	/// 2. Read parallel Mersenne Twister state for each kernel.
	this->_mts = prng::read_parallel_mts(MT_CONFIG_FILE, ntotal, this->_noseed);
	this->_zeta.resize(shape);
	bsc::upstream(
		this,
		new ar_master_kernel<T>(
			*this,
			partshape,
			nparts
		)
	);
}

template <class T>
void
arma::generator::AR_model<T>::react(bsc::kernel* child) {
//	ar_master_kernel<T>* master = dynamic_cast<ar_master_kernel<T>*>(child);
	bsc::commit(this);
}

template <class T>
arma::Array3D<T>
arma::generator::AR_model<T>::do_generate() {
	throw std::runtime_error("bad method");
}
