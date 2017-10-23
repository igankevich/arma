#include <thread>
#include <vector>

#if defined(ARMA_SIMULATE_FAILURES)
#include <ctsdlib>
#endif

#include <bscheduler/api.hh>

#include "config.hh"
#include "parallel_mt.hh"
#include "profile.hh"
#include "profile_counters.hh"
#include "util.hh"

#include "ar_model_wn.cc"

#include "bits/bscheduler_io.hh"

namespace {

	template <class T>
	class ar_partition_kernel: public bsc::kernel {

	private:
		typedef arma::Shape3D shape_type;
		typedef arma::Array3D<T> array_type;
		typedef blitz::RectDomain<3> rect_type;
		typedef arma::prng::parallel_mt generator_type;

	private:
		/// Partition start.
		shape_type _lower;
		/// Partition end.
		shape_type _upper;
		/// Part index in the counter array.
		shape_type _part;
		/// White noise variance.
		T _varwn = 0;
		/// Wavy surface part including all dependent points.
		array_type _zeta;
		/// AR coefficients.
		array_type _phi;
		/// Parallel Mersenne Twister.
		generator_type _generator;

	public:

		ar_partition_kernel() = default;

		inline
		ar_partition_kernel(
			shape_type lower,
			shape_type upper,
			shape_type part,
			T varwn,
			array_type zeta,
			array_type phi,
			const generator_type& generator
		):
		_lower(lower),
		_upper(upper),
		_part(part),
		_varwn(varwn),
		_zeta(zeta),
		_phi(phi),
		_generator(generator)
		{}

		void
		act() override {
			#if defined(ARMA_SIMULATE_FAILURES)
			if (const char* envvar = std::getenv("_FAILURE")) {
				using namespace bsc::this_application;
				std::string failure = envvar;
				if ((failure == "master" && is_master()) ||
					(failure == "slave" && is_slave())) {
					using namespace sys;
					send(signal::kill, this_process::parent_id());
					std::exit(this_process::execute_command("false"));
				}
			}
			#endif
			ARMA_EVENT_START("generate_surface", "bsc", 0);
			rect_type subpart = this->part_bounds();
			this->_zeta(subpart) =
				generate_white_noise(
					this->get_part_shape(),
					this->_varwn,
					std::ref(this->_generator)
				);
			ar_generate_surface(this->_zeta, this->_phi, subpart);
			ARMA_EVENT_END("generate_surface", "bsc", 0);
			bsc::commit<bsc::Remote>(this);
		}

		inline const shape_type&
		get_part_index() const noexcept {
			return this->_part;
		}

		inline const array_type&
		zeta() const noexcept {
			return this->_zeta;
		}

		/// Bounds of the partition in big zeta array.
		inline rect_type
		bounds() const noexcept {
			return rect_type(this->_lower, this->_upper);
		}

		/// Bounds of the partition inside small zeta array.
		inline rect_type
		part_bounds() const noexcept {
			shape_type size = this->_zeta.shape();
			shape_type offset = size - this->get_part_shape();
			return rect_type(offset, size-1);
		}

		void
		write(sys::pstream& out) const override {
			ARMA_PROFILE_CNT_START(CNT_BSC_MARSHALLING);
			bsc::kernel::write(out);
			out << this->_lower;
			out << this->_upper;
			out << this->_part;
			rect_type subpart = this->part_bounds();
			if (this->moves_upstream()) {
				out << this->_varwn;
				out << this->_zeta.shape();
				const shape_type& offset = subpart.lbound();
				using blitz::Range;
				using blitz::toEnd;
				if (offset(1) > 0 && offset(2) > 0) {
					out << array_type(this->_zeta(
						Range::all(),
						Range(0, offset(1)-1),
						Range(0, offset(2)-1)
					));
				}
				if (offset(0) > 0 && offset(2) > 0) {
					out << array_type(this->_zeta(
						Range(0, offset(0)-1),
						Range(offset(1), toEnd),
						Range(0, offset(2)-1)
					));
				}
				if (offset(0) > 0 && offset(1) > 0) {
					out << array_type(this->_zeta(
						Range(0, offset(0)-1),
						Range(0, offset(1)-1),
						Range(offset(1), toEnd)
					));
				}
				out << this->_phi;
				out << this->_generator;
			} else {
				out << this->_zeta.shape();
				out << array_type(this->_zeta(subpart));
			}
			ARMA_PROFILE_CNT_END(CNT_BSC_MARSHALLING);
		}

		void
		read(sys::pstream& in) override {
			ARMA_PROFILE_CNT_START(CNT_BSC_MARSHALLING);
			bsc::kernel::read(in);
			in >> this->_lower;
			in >> this->_upper;
			in >> this->_part;
			if (this->moves_upstream()) {
				in >> this->_varwn;
				shape_type zeta_shape;
				in >> zeta_shape;
				this->_zeta.resize(zeta_shape);
				rect_type subpart = this->part_bounds();
				const shape_type& offset = subpart.lbound();
				using blitz::Range;
				using blitz::toEnd;
				if (offset(1) > 0 && offset(2) > 0) {
					array_type tmp1(this->_zeta(
						Range::all(),
						Range(0, offset(1)-1),
						Range(0, offset(2)-1)
					));
					in >> tmp1;
				}
				if (offset(0) > 0 && offset(2) > 0) {
					array_type tmp2(this->_zeta(
						Range(0, offset(0)-1),
						Range(offset(1), toEnd),
						Range(0, offset(2)-1)
					));
					in >> tmp2;
				}
				if (offset(0) > 0 && offset(1) > 0) {
					array_type tmp3(this->_zeta(
						Range(0, offset(0)-1),
						Range(0, offset(1)-1),
						Range(offset(1), toEnd)
					));
					in >> tmp3;
				}
				in >> this->_phi;
				in >> this->_generator;
			} else {
				shape_type zeta_shape;
				in >> zeta_shape;
				this->_zeta.resize(zeta_shape);
				rect_type subpart = this->part_bounds();
				array_type tmp(this->_zeta(subpart));
				in >> tmp;
			}
			ARMA_PROFILE_CNT_END(CNT_BSC_MARSHALLING);
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
			arma::write_key_value(
				std::clog,
				"White noise variance",
				this->_varwn
			);
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
			ARMA_PROFILE_CNT_START(CNT_BSC_COPY);
			using blitz::all;
			slave_kernel* slave = dynamic_cast<slave_kernel*>(child);
			const shape_type& lower = slave->get_part_index();
			this->_model._zeta(slave->bounds()) =
				slave->zeta()(slave->part_bounds());
			arma::print_progress("generated part", ++this->_nfinished, this->_ntotal);
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
				send_subordinate(
					shape_type(
						lower(0)+1,
						lower(1)+1,
						lower(2)+
						1
					)
				);
			}
			ARMA_PROFILE_CNT_END(CNT_BSC_COPY);
		}

	private:

		inline void
		send_subordinate(const shape_type& part_index) {
			using blitz::any;
			using blitz::min;
			using blitz::max;
			using blitz::shape;
			if (any(part_index == this->_nparts)) {
				return;
			}
			if (++this->_counter(part_index) == 7) {
				// sys::log_message("ar", "part=_", part_index);
				// increment counter to prevent calculation
				// of the part multiple times
				++this->_counter(part_index);
				shape_type lower = this->_partshape*part_index;
				shape_type upper =
					min(lower + this->_partshape, this->zeta_shape()) - 1;
				// include all dependent points
				shape_type big_lower =
					max(lower - this->_model._phi.shape(), shape(0,0,0));
				rect_type big_rect(big_lower, upper);
				bsc::upstream<bsc::Remote>(
					this,
					new slave_kernel(
						lower,
						upper,
						part_index,
						this->_varwn,
						this->_model._zeta(big_rect),
						this->_model._phi,
						this->mersenne_twister()
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

	struct auto_register_types {
		auto_register_types() {
			bsc::register_type<ar_partition_kernel<ARMA_REAL_TYPE> >();
		}

	} __autoregister;

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
	bsc::commit<bsc::Remote>(this);
}

template <class T>
arma::Array3D<T>
arma::generator::AR_model<T>::do_generate() {
	throw std::runtime_error("bad method");
}

template <class T>
void
arma::generator::AR_model<T>
::write(sys::pstream& out) const {
	Basic_ARMA_model<T>::write(out);
	out << this->_partition;
	out << this->_phi;
	out << this->_doleastsquares;
}

template <class T>
void
arma::generator::AR_model<T>
::read(sys::pstream& in) {
	Basic_ARMA_model<T>::read(in);
	in >> this->_partition;
	in >> this->_phi;
	in >> this->_doleastsquares;
}
