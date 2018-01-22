#include "acf.hh"
#include "acf_wrapper.hh"
#include "params.hh"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

namespace {

	typedef std::function<arma::Array3D<ARMA_REAL_TYPE>(
							  const arma::Vec3D<ARMA_REAL_TYPE>&,
							  const arma::Shape3D&,
							  ARMA_REAL_TYPE,
							  ARMA_REAL_TYPE,
							  const arma::Vec3D<ARMA_REAL_TYPE>&,
							  const arma::Vec2D<ARMA_REAL_TYPE>&
	                      )> ACF_function;

	/// Map of names to ACF functions.
	const std::unordered_map<std::string, ACF_function>
	acf_functions = {
		{"standing_wave", arma::standing_wave_ACF<ARMA_REAL_TYPE>},
		{"propagating_wave", arma::propagating_wave_ACF<ARMA_REAL_TYPE>}
	};

	ACF_function
	get_acf_function(std::string func) {
		auto result = acf_functions.find(func);
		if (result == acf_functions.end()) {
			std::cerr << "Bad ACF function name: \"" << func << '\"' <<
			    std::endl;
			throw std::runtime_error("bad ACF function name");
		}
		return result->second;
	}

}

template <class T>
std::istream&
arma::bits::operator>>(std::istream& in, ACF_wrapper<T>& rhs) {
	std::string func;
	Grid<T,3> grid;
	sys::parameter_map params {
		{
			{"grid", sys::make_param(grid, validate_grid<T, 3>)},
			{"func", sys::make_param(func)},
			{"amplitude", sys::make_param(rhs._amplitude)},
			{"velocity", sys::make_param(rhs._velocity)},
			{"alpha", sys::make_param(rhs._alpha)},
			{"beta", sys::make_param(rhs._beta)},
		},
		true
	};
	in >> params;
	ACF_function acf_func = get_acf_function(func);
	rhs._acf.resize(grid.size());
	rhs._acf.reference(acf_func(
		grid.delta(),
		grid.size(),
		rhs._amplitude,
		rhs._velocity,
		rhs._alpha,
		rhs._beta
	));
	rhs._acf.setgrid(grid);
	return in;
}

template class arma::bits::ACF_wrapper<ARMA_REAL_TYPE>;

template std::istream&
arma::bits::operator>>(std::istream& in, ACF_wrapper<ARMA_REAL_TYPE>& rhs);
