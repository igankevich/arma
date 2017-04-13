#include "high_amplitude_realtime_solver.hh"
#if ARMA_OPENCL
#include "opencl/vec.hh"
#endif
#include "profile.hh"
#include "derivative.hh"

namespace {

#include "high_amplitude_realtime_solver_opencl.cc"

}

template <class T>
arma::Array4D<T>
arma::velocity::High_amplitude_realtime_solver<T>::operator()(
	const Discrete_function<T,3>& zeta
) {
	using opencl::context;
	using blitz::product;
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
	if (!this->_wfunc()) {
		this->_wfunc = buffer_type(
			context(),
			CL_MEM_READ_WRITE,
			product(grid.num_points())*sizeof(T)
		);
	}
	if (!this->_sfunc()) {
		this->_sfunc = buffer_type(
			context(),
			CL_MEM_READ_WRITE,
			zeta.numElements()*sizeof(T)
		);
	}
	ARMA_PROFILE_BLOCK("second_function",
		compute_second_function(zeta);
	);
	for (int i=0; i<nt; ++i) {
		ARMA_PROFILE_BLOCK("window_function::compute",
			compute_window_function(grid);
		);
		ARMA_PROFILE_BLOCK("fft",
			compute_velocity_field(grid);
		);
	}
	return Array4D<T>();
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
	const Discrete_function<T,3>& zeta
) {
/*
	using opencl::context;
	typedef opencl::Vec<Vector<T,3>,T,3> Vec3;
	const Vector<size_t,3> shp(zeta.shape());
	cl::Buffer bzeta(
		context(),
		const_cast<T*>(zeta.data()),
		const_cast<T*>(zeta.data() + zeta.numElements()),
		true
	);
	cl::Kernel kernel = opencl::get_kernel(__func__, HARTS_SRC);
	kernel.setArg(0, bzeta);
	kernel.setArg(1, Vec3(zeta.grid().length()));
	kernel.setArg(2, 0); // t derivative
	kernel.setArg(3, this->_sfunc);
	opencl::command_queue().enqueueNDRangeKernel(
		kernel,
		cl::NullRange,
		cl::NDRange(shp(0), shp(1), shp(2))
	);
*/
	const int nt = this->_domain.num_points(0);
	Array3D<T> zeta_t(zeta.shape());
	for (int i=0; i<nt; ++i) {
		using blitz::Range;
		zeta_t(i, Range::all(), Range::all()) = -derivative<0,T>(
			zeta,
			zeta.grid().delta(),
			i
		);
		cl::copy(
			opencl::command_queue(),
			zeta_t.data(),
			zeta_t.data() + zeta_t.numElements(),
			this->_sfunc
		);
	}
}


template <class T>
void
arma::velocity::High_amplitude_realtime_solver<T>::compute_velocity_field(
	const Grid<T,3>& domain
) {
//	typedef opencl::Vec<Vector<T,4>,T,4> Vec4;
//	const Vector<size_t,4> shp(nt, nz, nx, ny);
//	cl::Kernel kernel = opencl::get_kernel("compute_velocity_field", HARTS_SRC);
//	kernel.setArg(0, this->_phi);
//	opencl::command_queue().enqueueNDRangeKernel(
//		kernel,
//		cl::NullRange,
//		cl::NDRange(shp(1), shp(2), shp(3))
//	);
}

template class arma::velocity::High_amplitude_realtime_solver<ARMA_REAL_TYPE>;
