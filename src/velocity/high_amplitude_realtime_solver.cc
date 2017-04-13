#include "high_amplitude_realtime_solver.hh"
#if ARMA_OPENCL
#include "opencl/vec.hh"
#endif

namespace {

#include "high_amplitude_realtime_solver_opencl.cc"

}

template <class T>
arma::Array4D<T>
arma::velocity::High_amplitude_realtime_solver<T>::operator()(
	const Discrete_function<T,3>& zeta
) {
	using opencl::context;
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
		this->_wfunc = cl::BufferGL(context(), CL_MEM_READ_WRITE, 123);
	}
	compute_window_function(grid);
	typedef opencl::Vec<Vector<T,4>,T,4> Vec4;
	const Vector<size_t,4> shp(nt, nz, nx, ny);
	cl::Kernel kernel = opencl::get_kernel("compute_velocity_field", HARTS_SRC);
	kernel.setArg(0, this->_phi);
	opencl::command_queue().enqueueNDRangeKernel(
		kernel,
		cl::NullRange,
		cl::NDRange(shp(1), shp(2), shp(3))
	);
	return Array4D<T>();
}

template <class T>
void
arma::velocity::High_amplitude_realtime_solver<T>::compute_window_function(
	const Grid<T,3>& grid
) {
	typedef opencl::Vec<Vector<T,3>,T,3> Vec3;
	cl::Kernel kernel = opencl::get_kernel(__func__, HARTS_SRC);
	kernel.setArg(0, Vec3(grid.length()));
	kernel.setArg(1, this->_depth);
	kernel.setArg(2, this->_wfunc);
}

template class arma::velocity::High_amplitude_realtime_solver<ARMA_REAL_TYPE>;
