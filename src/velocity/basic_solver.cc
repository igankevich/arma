#include "basic_solver.hh"
#include "params.hh"
#include "validators.hh"
#include "interpolate.hh"
#include "profile.hh"
#if ARMA_PROFILE
#include "profile_counters.hh"
#endif

template<class T>
void
arma::velocity::Velocity_potential_solver<T>::write(std::ostream& out) const {
	out << "wnmax=" << this->_wnmax << ','
		<< "depth=" << this->_depth << ','
		<< "domain=" << this->_domain;
}

template <class T>
void
arma::velocity::Velocity_potential_solver<T>::read(std::istream& in) {
	using arma::validate_finite;
	using arma::validate_domain;
	sys::parameter_map params({
		{"wnmax", sys::make_param(_wnmax, validate_domain<T,2>)},
		{"depth", sys::make_param(_depth, validate_finite<T>)},
		{"domain", sys::make_param(_domain, validate_domain<T,2>)},
	}, true);
	in >> params;
}

template <class T>
arma::Array4D<T>
arma::velocity::Velocity_potential_solver<T>::operator()(
	const Discrete_function<T,3>& zeta
) {
	using blitz::Range;
	const Shape3D& zeta_size = zeta.shape();
	const Shape2D arr_size(zeta_size(1), zeta_size(2));
	const int nt = _domain.num_points(0);
	const int nz = _domain.num_points(1);
	const int nx = zeta_size(1);
	const int ny = zeta_size(2);
	Array4D<T> result(blitz::shape(nt, nz, nx, ny));
	precompute(zeta);
	for (int i=0; i<nt; ++i) {
		const T t = _domain(i, 0);
		ARMA_PROFILE_FUNC_CNT(CNT_SECONDFUNC, precompute(zeta, t));
		#if ARMA_OPENMP
		#pragma omp parallel for
		#endif
		for (int j=0; j<nz; ++j) {
			const T z = _domain(j, 1);
			Array2D<T> res = compute_velocity_field_2d(zeta, arr_size, z, t);
			ARMA_PROFILE_BLOCK_CNT(CNT_FFT, "interpolate",
				res(0,0) = interpolate({1,1}, {1,2}, {2,1}, res, {0,0});
				for (int k=1; k<nx; ++k) {
					res(k,0) = interpolate({k-1,1}, {k,1}, {k-1,2}, res, {k,0});
				}
				for (int l=1; l<ny; ++l) {
					res(0,l) = interpolate({1,l-1}, {1,l}, {2,l-1}, res, {0,l});
				}
				result(i, j, Range::all(), Range::all()) = res;
			);
		}
//		std::clog << "Finished time slice ["
//			<< (i+1) << '/' << nt << ']'
//			<< std::endl;
	}
	return result;
}

template class arma::velocity::Velocity_potential_solver<ARMA_REAL_TYPE>;
