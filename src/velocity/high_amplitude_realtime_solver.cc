#include "high_amplitude_realtime_solver.hh"
#if ARMA_OPENCL
#include "opencl/vec.hh"
#endif
#include "profile.hh"
#include "derivative.hh"

#include <complex>
#include <type_traits>

namespace {

#include "high_amplitude_realtime_solver_opencl.cc"

}

#define CHECK(x) ::cl::detail::errHandler((x), #x);

template <class T>
arma::velocity::High_amplitude_realtime_solver<T>::High_amplitude_realtime_solver() {
	CHECK(clfftInitSetupData(&this->_fft));
	CHECK(clfftSetup(&this->_fft));
}

template <class T>
arma::velocity::High_amplitude_realtime_solver<T>::~High_amplitude_realtime_solver() {
	CHECK(clfftDestroyPlan(&this->_fftplan));
	CHECK(clfftTeardown());
}

template <class T>
void
arma::velocity::High_amplitude_realtime_solver<T>::setup(
	const Discrete_function<T, 3>& zeta,
	const Grid<T, 3>& grid
) {
	using opencl::context;
	using opencl::command_queue;
	using blitz::product;
	if (!this->_wfunc()) {
		this->_wfunc = buffer_type(
			context(),
			CL_MEM_READ_WRITE,
			product(grid.num_points())*sizeof(T)
		);
	}
	if (!_phi()) {
		_phi = buffer_type(
			context(),
			CL_MEM_READ_WRITE,
			product(grid.num_points())*sizeof(T)*2
		);
	}
	Vector<size_t,2> lengths(grid.num_points(1), grid.num_points(2));
	CHECK(clfftCreateDefaultPlan(&_fftplan, context()(), CLFFT_2D, lengths.data()));
	clfftPrecision prec = std::conditional<
		std::is_same<T,float>::value,
		std::integral_constant<clfftPrecision,CLFFT_SINGLE>,
		std::integral_constant<clfftPrecision,CLFFT_DOUBLE>
	>::type::value;
	CHECK(clfftSetPlanPrecision(_fftplan, prec));
	CHECK(clfftSetLayout(
		_fftplan,
		CLFFT_COMPLEX_INTERLEAVED,
		CLFFT_COMPLEX_INTERLEAVED
	));
	CHECK(clfftSetResultLocation(_fftplan, CLFFT_INPLACE));
	CHECK(clfftSetPlanBatchSize(_fftplan, grid.num_points(0)));
	CHECK(clfftBakePlan(_fftplan, 1, &command_queue()(), nullptr, nullptr));
}

template <class T>
arma::Array4D<T>
arma::velocity::High_amplitude_realtime_solver<T>::operator()(
	const Discrete_function<T,3>& zeta
) {
	using blitz::shape;
	const Shape3D& zeta_size = zeta.shape();
	const Shape2D arr_size(zeta_size(1), zeta_size(2));
	const int nt = this->_domain.num_points(0);
	const int nz = this->_domain.num_points(1);
	const int nx = zeta_size(1);
	const int ny = zeta_size(2);
	const Grid<T,3> grid(
		{nz, nx, ny},
		{this->_domain.length(1), zeta.grid().length(1), zeta.grid().length(2)}
	);
	Array4D<T> result(shape(nt, nz, nx, ny));
	ARMA_PROFILE_BLOCK("setup",
		setup(zeta, grid);
	);
	for (int i=0; i<nt; ++i) {
		ARMA_PROFILE_BLOCK("second_function",
			compute_second_function(zeta, i);
		);
		ARMA_PROFILE_BLOCK("window_function::compute",
			compute_window_function(grid);
		);
		ARMA_PROFILE_BLOCK("fft",
			compute_velocity_field(grid);
		);
		const int stride_t = result.stride(0);
		std::clog << "stride_t=" << stride_t << std::endl;
		cl::copy(
			opencl::command_queue(),
			_phi,
			result.data() + i*stride_t,
			result.data() + (i+1)*stride_t
		);
	}
	return result;
}

template <class T>
void
arma::velocity::High_amplitude_realtime_solver<T>::compute_window_function(
	const Grid<T,3>& grid
) {
	typedef opencl::Vec<Vector<T,3>,T,3> Vec3;
	const Vector<size_t,3> shp(grid.num_points());
	cl::Kernel kernel = opencl::get_kernel(__func__, HARTS_SRC);
	kernel.setArg(0, Vec3(grid.length()));
	kernel.setArg(1, this->_depth);
	kernel.setArg(2, this->_wfunc);
	opencl::command_queue().enqueueNDRangeKernel(
		kernel,
		cl::NullRange,
		cl::NDRange(shp(0), shp(1), shp(2))
	);
}

template <class T>
void
arma::velocity::High_amplitude_realtime_solver<T>::compute_second_function(
	const Discrete_function<T,3>& zeta,
	const int idx_t
) {
	using blitz::shape;
	using blitz::Range;
	const int nz = this->_domain.num_points(1);
	const int nx = zeta.extent(1);
	const int ny = zeta.extent(2);
	Array3D<std::complex<T>> zeta_t(shape(nz, nx, ny));
	zeta_t(0, Range::all(), Range::all()) = -derivative<0,T>(
		zeta,
		zeta.grid().delta(),
		idx_t
	);
	for (int i=1; i<nz; ++i) {
		zeta_t(i, Range::all(), Range::all()) = zeta_t(0, Range::all(), Range::all());
	}
	cl::copy(
		opencl::command_queue(),
		zeta_t.data(),
		zeta_t.data() + zeta_t.numElements(),
		this->_phi
	);
}


template <class T>
void
arma::velocity::High_amplitude_realtime_solver<T>::compute_velocity_field(
	const Grid<T,3>& domain
) {
	using opencl::command_queue;
	const int nz = domain.num_points(0);
	const int nx = domain.num_points(1);
	const int ny = domain.num_points(2);
	const int stride_z = nx*ny;
	for (int i=0; i<nz; ++i) {
		cl_buffer_region region;
		region.origin = i*stride_z*sizeof(T);
		region.size = stride_z*sizeof(T);
		cl::Buffer phi_slice_z = _phi.createSubBuffer(
			CL_MEM_READ_WRITE,
			CL_BUFFER_CREATE_TYPE_REGION,
			&region
		);
		CHECK(clfftEnqueueTransform(
			_fftplan,
			CLFFT_FORWARD,
			1,
			&command_queue()(),
			0,
			nullptr,
			nullptr,
			&phi_slice_z(),
			nullptr,
			nullptr
		));
	}
	command_queue().finish();
	std::clog << "finish" << std::endl;
}

template class arma::velocity::High_amplitude_realtime_solver<ARMA_REAL_TYPE>;

#undef CHECK
