#include "waves.hh"

#include <ostream>
#include <stdexcept>

#include "apmath/convolution.hh"
#include "physical_constants.hh"
#include "statistics.hh"

namespace {

	template <class T>
	arma::Array1D<T>
	to_waves(const std::vector<T>& rhs) {
		arma::Array1D<T> lhs(rhs.size());
		std::copy_n(rhs.data(), rhs.size(), lhs.data());
		return lhs;
	}

	template <class C1, class C2>
	void
	push_back_all(C1& lhs, const C2& rhs) {
		lhs.insert(lhs.end(), rhs.begin(), rhs.end());
	}

	template <class T>
	arma::Array1D<arma::stats::Wave<T>>
	extract_waves(
		arma::Array3D<T> elevation,
		const arma::Grid<T,3>& grid,
		int dimension,
		int kradius
	) {
		using blitz::Range;
		using arma::Domain;
		using arma::stats::Wave;
		using arma::stats::find_waves;
		const int nt = elevation.extent(0);
		const int nx = elevation.extent(1);
		const int ny = elevation.extent(2);
		Domain<T,1> grid1d {
			{T(0)},
			{grid.length(dimension)},
			{grid.num_points(dimension)}
		};
		std::vector<Wave<T>> result;
		if (dimension == 0) {
			for (int i = 0; i < nx; ++i) {
				for (int j = 0; j < ny; ++j) {
					push_back_all(
						result,
						find_waves(
							elevation(Range::all(), i, j),
							grid1d,
							kradius
						)
					);
				}
			}
		} else if (dimension == 1) {
			for (int i = 0; i < nt; ++i) {
				for (int j = 0; j < ny; ++j) {
					push_back_all(
						result,
						find_waves(
							elevation(i, Range::all(), j),
							grid1d,
							kradius
						)
					);
				}
			}
		} else if (dimension == 2) {
			for (int i = 0; i < nt; ++i) {
				for (int j = 0; j < nx; ++j) {
					push_back_all(
						result,
						find_waves(
							elevation(i, j, Range::all()),
							grid1d,
							kradius
						)
					);
				}
			}
		} else {
			throw std::invalid_argument("bad dimension");
		}
		return to_waves(result);
	}

}


template <class T>
arma::stats::Wave_field<T>
::Wave_field(
	Array3D<T> elevation,
	const Grid<T, 3>& grid,
	int kradius
):
_wavest(extract_waves(elevation, grid, 0, kradius)),
_wavesx(extract_waves(elevation, grid, 1, kradius)),
_wavesy(extract_waves(elevation, grid, 2, kradius))
{}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>
::lengths() const {
	using blitz::Range;
	using blitz::toEnd;
	const int nx = this->_wavesx.size();
	const int ny = this->_wavesy.size();
	Array1D<T> result(nx+ny);
	result(Range(0, nx-1)) = this->_wavesx[Wave<T>::clength];
	result(Range(nx, toEnd)) = this->_wavesy[Wave<T>::clength];
	return result;
}

template <class T>
arma::Array1D<T>
arma::stats::Wave_field<T>
::heights() const {
	using blitz::Range;
	using blitz::toEnd;
	const int nx = this->_wavesx.size();
	const int ny = this->_wavesy.size();
	Array1D<T> result(nx+ny);
	result(Range(0, nx-1)) = this->_wavesx[Wave<T>::cheight];
	result(Range(nx, toEnd)) = this->_wavesy[Wave<T>::cheight];
	return result;
}

template <class T>
std::vector<arma::stats::Wave_feature<T>>
arma::stats
::find_extrema(Array1D<T> elevation, const Domain<T,1>& grid) {
	std::vector<Wave_feature<T>> result;
	const int n = elevation.numElements();
	if (n < 3) {
		return result;
	}
	for (int i=1; i<n-1; ++i) {
		const T x0 = grid(i-1, 0);
		const T x1 = grid(i, 0);
		const T x2 = grid(i+1, 0);
		const T z0 = elevation(i-1);
		const T z1 = elevation(i);
		const T z2 = elevation(i+1);
		if ((z1 > z0 && z1 > z2) || (z1 < z0 && z1 < z2)) {
			// approximate three points with a parabola
			const T denominator = ((x0-x1) * (x0-x2) * (x1-x2));
			const T a = (x0*(z2-z1) + x1*(z0-z2) + x2*(z1-z0)) /
			            denominator;
			const T b = (x0*x0*(z1-z2) + x1*x1*(z2-z0) + x2*x2*(z0-z1)) /
			            denominator;
			const T c = (z0*(x1*x1*x2 - x1*x2*x2) +
			             z1*(x0*x2*x2 - x0*x0*x2) +
			             z2*(x0*x0*x1 - x0*x1*x1)) / denominator;
			// find parabola vertex
			const T vx = -b / (T(2)*a);
			const T vz = -(b*b - T(4)*a*c) / (T(4)*a);
			const Wave_feature_type type =
				z1 > z0 ? Wave_feature_type::Crest : Wave_feature_type::Trough;
			result.emplace_back(vx, vz, type);
		}
	}
	return result;
}

template <class T>
std::vector<arma::stats::Wave<T>>
arma::stats
::find_waves(const std::vector<Wave_feature<T>>& features) {
	using std::abs;
	std::vector<Wave<T>> result;
	if (features.empty()) {
		return result;
	}
	const int n = features.size();
	Wave_feature<T> f0 = features[0];
	for (int i=1; i<n; ++i) {
		Wave_feature<T> fi = features[i];
		if (fi.type != f0.type) {
			const T length = T(2)*abs(f0.x - fi.x);
			const T height = abs(f0.z - fi.z);
			result.emplace_back(height, length);
		}
		f0 = fi;
	}
	return result;
}

template <class T>
std::vector<arma::stats::Wave<T>>
arma::stats
::find_waves(Array1D<T> elevation, Domain<T,1> grid, int r) {
	Array1D<T> elevation_copy(elevation.copy());
	smooth_elevation(elevation_copy, grid, r);
	return find_waves(find_extrema(elevation_copy, grid));
}

template <class T>
std::ostream&
arma::stats::operator<<(std::ostream& out, const Wave_feature<T>& rhs) {
	return out << '(' << rhs.x << ',' << rhs.z << ',' << int(rhs.type) << ')';
}

template <class T>
arma::Array1D<T>
arma::stats
::gaussian_kernel(int r, T sigma) {
	using blitz::sum;
	using std::exp;
	const int n = r+r+1;
	const T denominator = T(2)*sigma*sigma;
	Domain<T,1> grid {{T(-r)}, {T(r)}, {n}};
	Array1D<T> result(n);
	for (int i=0; i<n; ++i) {
		const T x = grid(i,0);
		result(i) = exp(-x*x/denominator);
	}
	result /= sum(result);
	return result;
}

template <class T, int N>
blitz::Array<T,N>
arma::stats
::filter(blitz::Array<T,N> data, blitz::Array<T,N> kernel0) {
	typedef std::complex<T> C;
	typedef blitz::Array<C,N> array_type;
	array_type signal(data.shape());
	signal = data;
	array_type kernel(kernel0.shape());
	kernel = kernel0;
	apmath::Convolution<C,N> conv(signal, kernel);
	return blitz::real(conv.convolve(signal, kernel)).copy();
}

template <class T>
void
arma::stats
::smooth_elevation(Array1D<T>& elevation, Domain<T,1>& grid, int r) {
	using blitz::scale;
	arma::Array1D<T> new_elevation =
		filter(elevation, gaussian_kernel<T>(r));
	const T s = scale(elevation) / scale(new_elevation);
	elevation = new_elevation * s;
	grid.translate({-r});
}

template <class T>
arma::Array3D<T>
arma::stats::frequency_amplitude_spectrum(Array3D<T> rhs, const Grid<T,3>& grid) {
	using arma::apmath::Fourier_transform;
	using blitz::RectDomain;
	using blitz::abs;
	using blitz::product;
	using arma::constants::sqrt2pi;
	typedef std::complex<T> C;
	Array3D<C> rhs_copy(rhs.shape());
	rhs_copy = rhs;
	Fourier_transform<C,3> fft(rhs.shape());
	fft.forward(rhs_copy);
	const int n = rhs.numElements();
	const RectDomain<3> domain(rhs.shape()/2, rhs.shape()-1);
	return Array3D<T>(T(2) * abs(rhs_copy(domain)) / n);
}

template class arma::stats::Wave<ARMA_REAL_TYPE>;
template class arma::stats::Wave_field<ARMA_REAL_TYPE>;

template std::vector<arma::stats::Wave_feature<ARMA_REAL_TYPE>>
arma::stats
::find_extrema(
	Array1D<ARMA_REAL_TYPE> elevation,
	const Domain<ARMA_REAL_TYPE, 1>& grid
);

template std::vector<arma::stats::Wave<ARMA_REAL_TYPE>>
arma::stats
::find_waves(const std::vector<Wave_feature<ARMA_REAL_TYPE>>& features);

template std::vector<arma::stats::Wave<ARMA_REAL_TYPE>>
arma::stats
::find_waves(
	Array1D<ARMA_REAL_TYPE> elevation,
	Domain<ARMA_REAL_TYPE, 1> grid,
	int r
);

template arma::Array1D<ARMA_REAL_TYPE>
arma::stats
::gaussian_kernel(int r, ARMA_REAL_TYPE sigma);

template blitz::Array<ARMA_REAL_TYPE,1>
arma::stats
::filter(
	blitz::Array<ARMA_REAL_TYPE, 1> data,
	blitz::Array<ARMA_REAL_TYPE, 1> kernel
);

template void
arma::stats
::smooth_elevation(
	Array1D<ARMA_REAL_TYPE>& elevation,
	Domain<ARMA_REAL_TYPE, 1>& grid,
	int r
);

template std::ostream&
arma::stats::operator<<(
	std::ostream& out,
	const Wave_feature<ARMA_REAL_TYPE>& rhs
);

template arma::Array3D<ARMA_REAL_TYPE>
arma::stats::frequency_amplitude_spectrum(
	Array3D<ARMA_REAL_TYPE> rhs,
	const Grid<ARMA_REAL_TYPE, 3>& grid
);
