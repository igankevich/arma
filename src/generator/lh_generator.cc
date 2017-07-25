#include "lh_generator.hh"
#include "physical_constants.hh"
#include "params.hh"
#if ARMA_OPENCL
#include "opencl/opencl.hh"
#include "opencl/vec.hh"
#endif

#include <cmath>
#include <random>
#include <functional>
#include <atomic>
#include <fstream>

template<class T>
T
arma::generator::Longuet_Higgins_model<T>::approx_spectrum(T w, T theta, T h) {
	using std::pow;
	using std::cos;
	using std::exp;
	using std::sqrt;
	const int k = 5;
	const int n = 4;
	const T tau = T(4.8)*sqrt(h); // average wave period
	const T   A = T(0.28)*pow(T(2)*M_PI, 4)*h*h*pow(tau, (-n));
	const T   B = T(0.44)*pow(T(2)*M_PI, n)*pow(tau, (-n));
	return A*(pow(w, -k))*exp(-B*(pow(w,-n)))*T(2)*pow(cos(theta), 2)/M_PI;
	//*pow(cos(theta), n)/pow(2, n)*(pow(tgamma(0.5*(n+1)), 2)/tgamma(n+1));//
}

template<class T>
arma::Array2D<T>
arma::generator::Longuet_Higgins_model<T>::approximate_spectrum(
	const Domain<T,2>& domain,
	T wave_height
) {
	const int nomega = domain.num_points(0);
	const int ntheta = domain.num_points(1);
	Array2D<T> result(domain.num_points());
	#if ARMA_OPENMP
	#pragma omp parallel for collapse(2)
	#endif
	for (int i=0; i<nomega; ++i) {
		for (int j=0; j<ntheta; ++j) {
			const T w = domain(i, 0);
			const T theta = domain(j, 1);
			result(i, j) = approx_spectrum(w, theta, wave_height);
		}
	}
	return result;
}

template<class T>
arma::Array2D<T>
arma::generator::Longuet_Higgins_model<T>::determine_coefficients(
	const Domain<T,2>& sdom,
	T wave_height
) {
	using std::sqrt;
	using blitz::product;
	Array2D<T> result(sdom.num_points());
	const int nomega = sdom.num_patches(0);
	const int ntheta = sdom.num_patches(1);
	const int nomega2 = _spec_subdomain(0);
	const int ntheta2 = _spec_subdomain(1);
	#if ARMA_OPENMP
	#pragma omp parallel for collapse(2)
	#endif
	for (int i=0; i<nomega; ++i) {
		for (int j=0; j<ntheta; ++j) {
			const Domain<T,2> subdom(
				sdom({i,j}),
				sdom({i+1,j+1}),
				_spec_subdomain
			);
			T sum = 0;
			#if ARMA_OPENMP
			#pragma omp simd
			#endif
			for (int x=0; x<nomega2; x++) {
				for (int y=0; y<ntheta2; y++) {
					const T omega = subdom(i, 0);
					const T theta = subdom(j, 1);
					sum += approx_spectrum(omega, theta, wave_height);
				}
			}
			result(i, j) = sqrt(T(2)*sum*product(subdom.delta()));
		}
	}
	return result;
}

#if ARMA_OPENCL

template <class T>
void
arma::generator::Longuet_Higgins_model<T>::generate_surface(
	Discrete_function<T,3>& zeta,
	const Domain3D& subdomain
) {
	using opencl::context;
	using opencl::command_queue;
	using opencl::get_kernel;
	typedef opencl::Vec<Vec2D<T>,T,2> Vec2;
	typedef opencl::Vec<Vec2D<int>,int,2> Int2;
	typedef opencl::Vec<Vec3D<T>,T,3> Vec3;
	Vec3D<size_t> shp = zeta.shape();
	cl::Buffer bcoef(
		context(),
		_coef.data(),
		_coef.data() + _coef.numElements(),
		true
	);
	cl::Buffer beps(
		context(),
		_eps.data(),
		_eps.data() + _eps.numElements(),
		true
	);
	cl::Buffer bzeta(
		context(),
		CL_MEM_WRITE_ONLY,
		zeta.numElements()*sizeof(T)
	);
	cl::Kernel kernel = get_kernel("lh_generate_surface");
	kernel.setArg(0, bcoef);
	kernel.setArg(1, beps);
	kernel.setArg(2, bzeta);
	kernel.setArg(3, Vec2(_spec_domain.lbound()));
	kernel.setArg(4, Vec2(_spec_domain.ubound()));
	kernel.setArg(5, Int2(_spec_domain.num_patches()));
	kernel.setArg(6, Vec3(_outgrid.length()));
	command_queue().enqueueNDRangeKernel(
		kernel,
		cl::NullRange,
		cl::NDRange(shp(0), shp(1), shp(2))
	);
	cl::copy(
		command_queue(),
		bzeta,
		zeta.data(),
		zeta.data() + zeta.numElements()
	);
}

#else

template <class T>
void
arma::generator::Longuet_Higgins_model<T>::generate_surface(
	Discrete_function<T,3>& zeta,
	const Domain3D& subdomain
) {
	using std::cos;
	using std::sin;
	using constants::g;
	using blitz::product;
	const Shape3D& lbound = subdomain.lbound();
	const Shape3D& ubound = subdomain.ubound();
	const int i0 = lbound(0);
	const int j0 = lbound(1);
	const int k0 = lbound(2);
	const int i1 = ubound(0);
	const int j1 = ubound(1);
	const int k1 = ubound(2);
	const int nomega = _spec_domain.num_patches(0);
	const int ntheta = _spec_domain.num_patches(1);
	std::atomic<int> counter(0);
	const int total_count = product(ubound - lbound + 1);
	const int time_slice_count = total_count / (i1-i0+1);
	#if ARMA_OPENMP
	#pragma omp parallel for collapse(3)
	#endif
	for (int i=i0; i<=i1; ++i) {
		for (int j=j0; j<=j1; ++j) {
			for (int k=k0; k<=k1; ++k) {
				const T t = _outgrid(i, 0);
				const T x = _outgrid(j, 1);
				const T y = _outgrid(k, 2);
				T sum = 0;
				for (int l=0; l<nomega; ++l) {
					for (int m=0; m<ntheta; ++m) {
						const T omega = _spec_domain(l, 0);
						const T theta = _spec_domain(m, 1);
						const T omega_squared = omega*omega;
						const T k_x = omega_squared*cos(theta)/g<T>;
						const T k_y = omega_squared*sin(theta)/g<T>;
						sum += _coef(l,m)*cos(
							k_x*x + k_y*y - omega*t + _eps(l,m)
						);
					}
				}
				zeta(i,j,k) = sum;
				const int nfinished = ++counter;
				if ((nfinished % time_slice_count) == 0) {
					const int t0 = nfinished / time_slice_count;
					const int t1 = total_count / time_slice_count;
					std::clog << "Finished "
						<< '[' << t0 << '/' << t1 << ']' << std::endl;
				}
			}
		}
	}
}
#endif

template <class T>
void
arma::generator::Longuet_Higgins_model<T>::determine_coefficients() {
	_coef.reference(determine_coefficients(_spec_domain, _waveheight));
}

template <class T>
void
arma::generator::Longuet_Higgins_model<T>::generate_white_noise() {
	using constants::_2pi;
	_eps.resize(_spec_domain.num_points());
	std::mt19937 generator;
	std::uniform_real_distribution<T> dist(T(0), _2pi<T>);
	std::generate(
		std::begin(_eps),
		std::end(_eps),
		std::bind(dist, generator)
	);
}

template <class T>
void
arma::generator::Longuet_Higgins_model<T>::generate(
	Discrete_function<T,3>& zeta,
	const Domain3D& subdomain
) {
	generate_surface(zeta, subdomain);
}

template <class T>
std::istream&
arma::generator::operator>>(std::istream& in, Longuet_Higgins_model<T>& rhs) {
	sys::parameter_map params({
		{"spec_domain", sys::make_param(rhs._spec_domain)},
		{"spec_subdomain", sys::make_param(rhs._spec_subdomain)},
		{"wave_height", sys::make_param(rhs._waveheight)},
	}, true);
	in >> params;
	validate_domain<T,2>(rhs._spec_domain, "lh_model.spec_domain");
	validate_shape(rhs._spec_subdomain, "lh_model.spec_subdomain");
	validate_positive(rhs._waveheight, "lh_model.wave_height");
	return in;
}

template <class T>
std::ostream&
arma::generator::operator<<(
	std::ostream& out,
	const Longuet_Higgins_model<T>& rhs
) {
	return out
		<< "spec_domain=" << rhs._spec_domain
		<< ",spec_subdomain=" << rhs._spec_subdomain
		<< ",wave_height=" << rhs._waveheight;
}

template class arma::generator::Longuet_Higgins_model<ARMA_REAL_TYPE>;

template std::istream&
arma::generator::operator>>(
	std::istream& in,
	Longuet_Higgins_model<ARMA_REAL_TYPE>& rhs
);

template std::ostream&
arma::generator::operator<<(
	std::ostream& out,
	const Longuet_Higgins_model<ARMA_REAL_TYPE>& rhs
);
