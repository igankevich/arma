#include "nit_transform.hh"
#include "params.hh"
#include "validators.hh"
#include "transforms.hh"
#include "series.hh"
#include "util.hh"
#if ARMA_OPENCL
#include "opencl/opencl.hh"
#include "opencl/vec.hh"
#endif

#include <string>
#include <stdexcept>
#include <iostream>
#include <cmath>

namespace {

	#if ARMA_OPENCL
	template <class T, class Solver, class Distribution>
	void
	do_transform_realisation_opencl(
		arma::Array3D<T> acf,
		arma::Array3D<T>& realisation,
		const Solver& solver,
		const Distribution& dist
	) {
		using namespace arma;
		cl::Kernel kernel = opencl::get_kernel("transform_data_gram_charlier");
		kernel.setArg(0, realisation.buffer());
		kernel.setArg(1, solver.interval().first());
		kernel.setArg(2, solver.interval().last());
		kernel.setArg(3, solver.num_iterations());
		kernel.setArg(4, std::sqrt(acf(0,0,0)));
		kernel.setArg(5, dist.skewness());
		kernel.setArg(6, dist.kurtosis());
		realisation.compute(kernel);
	}
	#endif

}

template <class T>
void
arma::nonlinear::NIT_transform<T>::transform_CDF(Array3D<T> acf) {
	const T stdev = std::sqrt(acf(0,0,0));
	const auto& iv = this->_cdfsolver.interval();
	const Domain<T,1> grid(
		Vec1D<T>(iv.first()),
		Vec1D<T>(iv.last()),
		Vec1D<int>(this->_intnodes)
	);
	auto nodes = do_transform_CDF(stdev, grid);
	this->_xnodes.reference(nodes.first);
	this->_ynodes.reference(nodes.second);
}

template <class T>
std::pair<arma::Array1D<T>,arma::Array1D<T>>
arma::nonlinear::NIT_transform<T>::do_transform_CDF(
	const T stdev,
	const Domain<T,1>& grid
) {
	switch (_targetdist) {
		case bits::Distribution::Gram_Charlier:
			return
				::arma::nonlinear::transform_CDF(
					grid,
					normaldist_type(T(0), stdev),
					_gramcharlier,
					_cdfsolver
				);
		case bits::Distribution::Skew_normal:
			return
				::arma::nonlinear::transform_CDF(
					grid,
					normaldist_type(T(0), stdev),
					_skewnormal,
					_cdfsolver
				);
	}
	return std::make_pair(arma::Array1D<T>(),arma::Array1D<T>());
}

template <class T>
void
arma::nonlinear::NIT_transform<T>::transform_ACF(Array3D<T>& acf) {
	transform_CDF(acf);
	T err = std::numeric_limits<T>::max();
	Array1D<T> intcoefs;
	Array1D<T> gcscoefs;
	// find the optimal interpolation order
	for (unsigned int i=1; i<this->_maxintorder; ++i) {
		Array1D<T> new_intcoefs = linalg::interpolate(
			this->_xnodes,
			this->_ynodes,
			i
		);
		T new_err = std::numeric_limits<T>::max();
		Array1D<T> new_gcscoefs =
			gram_charlier_expand(
				new_intcoefs,
				this->_maxexpansionorder,
				acf(0,0,0),
				new_err
			);
		if (new_err < err) {
			intcoefs.resize(new_intcoefs.size());
			intcoefs = new_intcoefs;
			gcscoefs.resize(new_gcscoefs.size());
			gcscoefs = new_gcscoefs;
			err = new_err;
		}
	}
	#ifndef NDEBUG
	write_key_value(std::clog, "Optimal interpolation order", intcoefs.size());
	write_key_value(std::clog, "GCS approximation error", err);
	#endif
	::arma::nonlinear::transform_ACF(
		acf.data(),
		acf.numElements(),
		gcscoefs,
		this->_acfsolver
	);
}

template <class T>
void
arma::nonlinear::NIT_transform<T>::transform_realisation(
	Array3D<T> acf,
	Array3D<T>& realisation
) {
	switch (_targetdist) {
		case bits::Distribution::Gram_Charlier:
			#if ARMA_OPENCL
			do_transform_realisation_opencl(
				acf,
				realisation,
				this->_cdfsolver,
				this->_gramcharlier
			);
			#else
			do_transform_realisation(acf, realisation, this->_gramcharlier);
			#endif
			break;
		case bits::Distribution::Skew_normal:
			do_transform_realisation(acf, realisation, this->_skewnormal);
			break;
	}
}

template <class T>
template <class Dist>
void
arma::nonlinear::NIT_transform<T>::do_transform_realisation(
	Array3D<T> acf,
	Array3D<T>& realisation,
	Dist& dist
) {
	const T stdev = std::sqrt(acf(0,0,0));
	transform_data(
		realisation.data(),
		realisation.numElements(),
		normaldist_type(T(0), stdev),
		dist,
		this->_cdfsolver
	);
}

template <class T>
void
arma::nonlinear::NIT_transform<T>::read_dist(std::istream& str) {
	str >> _targetdist;
	switch (_targetdist) {
		case bits::Distribution::Gram_Charlier:
			str >> _gramcharlier;
			break;
		case bits::Distribution::Skew_normal:
			str >> _skewnormal;
			break;
		default:
			throw std::runtime_error("bad distribution");
	}
}

std::istream&
arma::nonlinear::bits::operator>>(std::istream& in, Distribution& rhs) {
	std::string name;
	in >> std::ws >> name;
	if (name == "gram_charlier") {
		rhs = Distribution::Gram_Charlier;
	} else if (name == "skew_normal") {
		rhs = Distribution::Skew_normal;
	} else {
		in.setstate(std::ios::failbit);
		std::cerr << "Invalid distribution: " << name << std::endl;
		throw std::runtime_error("bad distribution");
	}
	return in;
}

const char*
arma::nonlinear::bits::to_string(Distribution rhs) {
	switch (rhs) {
		case Distribution::Gram_Charlier: return "gram_charlier";
		case Distribution::Skew_normal: return "skew_normal";
		default: return "UNKNOWN";
	}
}

std::ostream&
arma::nonlinear::bits::operator<<(std::ostream& out, const Distribution& rhs) {
	return out << to_string(rhs);
}

template <class T>
std::ostream&
arma::nonlinear::operator<<(std::ostream& out, const NIT_transform<T>& rhs) {
	out << "dist=" << rhs._targetdist << ',';
	switch (rhs._targetdist) {
		case bits::Distribution::Gram_Charlier:
			out << rhs._gramcharlier;
			break;
		case bits::Distribution::Skew_normal:
			out << rhs._skewnormal;
			break;
	}
	out << ",cdf_solver=" << rhs._cdfsolver
		<< ",acf_solver=" << rhs._acfsolver
		<< ",interpolation_nodes=" << rhs._intnodes
		<< ",max_interpolation_order=" << rhs._maxintorder
		<< ",max_expansion_order=" << rhs._maxexpansionorder
		;
	return out;
}

template <class T>
std::istream&
arma::nonlinear::operator>>(std::istream& in, NIT_transform<T>& rhs) {
	sys::parameter_map::read_param
	read_dist = [&rhs] (std::istream& str, const char*) -> std::istream& {
		rhs.read_dist(str);
		return str;
	};
	sys::parameter_map params({
	    {"distribution", read_dist},
	    {
			"max_interpolation_order",
			sys::make_param(rhs._maxintorder, validate_positive<T>)
	    },
	    {
			"interpolation_nodes",
			sys::make_param(rhs._intnodes, validate_positive<T>)
	    },
	    {
			"max_expansion_order",
			sys::make_param(rhs._maxexpansionorder, validate_positive<T>)
	    },
	    {"cdf_solver", sys::make_param(rhs._cdfsolver)},
	    {"acf_solver", sys::make_param(rhs._acfsolver)},
	}, "nit_transform", true);
	in >> params;
	return in;
}

template class arma::nonlinear::NIT_transform<ARMA_REAL_TYPE>;

template std::ostream&
arma::nonlinear::operator<<(
	std::ostream& out,
	const NIT_transform<ARMA_REAL_TYPE>& rhs
);

template std::istream&
arma::nonlinear::operator>>(
	std::istream& in,
	NIT_transform<ARMA_REAL_TYPE>& rhs
);
