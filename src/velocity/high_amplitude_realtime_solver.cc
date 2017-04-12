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
	typedef opencl::Vec<Vector<T,4>,T,4> Vec4;
	const Shape3D& zeta_size = zeta.shape();
	const Shape2D arr_size(zeta_size(1), zeta_size(2));
	const int nt = this->_domain.num_points(0);
	const int nz = this->_domain.num_points(1);
	const int nx = zeta_size(1);
	const int ny = zeta_size(2);
	const Vector<size_t,4> shp(nt, nz, nx, ny);
	cl::Kernel kernel = opencl::get_kernel("compute_velocity_field", HARS_SRC);
	kernel.setArg(0, this->_phi);
	opencl::command_queue().enqueueNDRangeKernel(
		kernel,
		cl::NullRange,
		cl::NDRange(shp(1), shp(2), shp(3))
	);
	return Array4D<T>();
}

template class arma::velocity::High_amplitude_realtime_solver<ARMA_REAL_TYPE>;
