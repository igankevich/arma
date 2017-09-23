#include "delaunay_interpolation.hh"

#include <iterator>
#include <vector>

template <class T>
T
arma::Delaunay_interpolation<T>::
operator()(T x, T y) {
	typedef CGAL::Data_access<container_type> Value_access;
	const point_type p(x, y);
	std::vector< std::pair<point_type, T> > coords;
	T norm =
		CGAL::natural_neighbor_coordinates_2(
			this->_points,
			p,
			std::back_inserter(coords)
		).second;
	return CGAL::linear_interpolation(
		coords.begin(),
		coords.end(),
		norm,
		Value_access(this->_values)
	);
}

template class arma::Delaunay_interpolation<ARMA_REAL_TYPE>;
