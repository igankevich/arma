#include <random>
#include <vector>
#include <atomic>
#include "util.hh"
#include "opencl/opencl.hh"
#include "opencl/vec.hh"
#include "opencl/array.hh"
#include "blitz.hh"
#include "profile.hh"

namespace {

	struct progress_type {

		explicit
		progress_type(int n):
		second(n)
		{}

		std::atomic<int> first;
		int second;
	};

	void kernel_callback(cl_event ev, cl_int ret, void* user_data) {
		#if ARMA_PROFILE
		using namespace std::chrono;
		cl::Event event(ev);
		clRetainEvent(ev);
		cl_ulong t0 = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		cl_ulong t1 = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
		auto us0 = duration_cast<microseconds>(nanoseconds(t0));
		auto us1 = duration_cast<microseconds>(nanoseconds(t1));
		#endif
		progress_type* progress = static_cast<progress_type*>(user_data);
		const int nfinished = ++progress->first;
		const int ntotal = progress->second;
		std::clog << "finished " "[" << nfinished << '/' << ntotal << "]"
			#if ARMA_PROFILE
			" in " << (us1-us0).count() << "us"
			#endif
			<< '\n';
	}

}

template <class T>
arma::Array3D<T>
arma::generator::AR_model<T>::do_generate() {
	/// 1. Generate white noise on host.
	ARMA_PROFILE_START(generate_white_noise);
	Array3D<T> zeta = this->generate_white_noise();
	ARMA_PROFILE_END(generate_white_noise);
	using opencl::context;
	using opencl::command_queue;
	using opencl::get_kernel;
	using opencl::devices;
	typedef opencl::Vec<Vec3D<int>,int,3> Int3;
	/// 2. Partition the data.
	const Shape3D shape = this->_outgrid.size();
	const Shape3D partshape = get_partition_shape(
		this->_partition,
		this->grid().num_points(),
		this->order(),
		devices()[0].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()
	);
	const Shape3D nparts = blitz::div_ceil(shape, partshape);
	progress_type progress(product(nparts));
	write_key_value(std::clog, "Partition size", partshape);
	/// 3. Launch parallel kernels with dependencies
	/// controlled by OpenCL events.
	this->_phi.copy_to_device();
	zeta.copy_to_device();
	const int nt = nparts(0);
	const int nx = nparts(1);
	const int ny = nparts(2);
	Array3D<cl::Event> part_events(nparts);
	cl::Kernel kernel = get_kernel("ar_generate_surface");
	for (int i=0; i<nt; ++i) {
		for (int j=0; j<nx; ++j) {
			for (int k=0; k<ny; ++k) {
				const Shape3D ijk(i, j, k);
				const Shape3D lower = blitz::min(ijk * partshape, shape);
				const Shape3D upper = blitz::min((ijk+1) * partshape, shape) - 1;
				kernel.setArg(0, this->_phi.buffer());
				kernel.setArg(1, Int3(this->_phi.shape()));
				kernel.setArg(2, zeta.buffer());
				kernel.setArg(3, Int3(zeta.shape()));
				kernel.setArg(4, Int3(lower));
				kernel.setArg(5, Int3(upper));
				// add all already enqueued adjacent kernels
				// as dependencies
				std::vector<cl::Event> deps;
				const int m1 = std::max(i-1, 0);
				const int m2 = std::max(j-1, 0);
				const int m3 = std::max(k-1, 0);
				for (int l=m1; l<=i; ++l) {
					for (int m=m2; m<=j; ++m) {
						for (int n=m3; n<=k; ++n) {
							deps.push_back(part_events(l,m,n));
						}
					}
				}
				// remove the last invalid event
				deps.pop_back();
				command_queue().enqueueTask(
					kernel,
					&deps,
					&part_events(i,j,k)
				);
				part_events(i,j,k).setCallback(
					CL_COMPLETE,
					kernel_callback,
					&progress
				);
			}
		}
	}
	std::vector<cl::Event> all_events(part_events.begin(), part_events.end());
	cl::Event::waitForEvents(all_events);
	return zeta;
}



