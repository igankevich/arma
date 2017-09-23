#ifndef LINEAR_INTERPOLATON_HH
#define LINEAR_INTERPOLATON_HH

#include <map>

#include <CGAL/number_utils.h>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/interpolation_functions.h>
#include <CGAL/natural_neighbor_coordinates_2.h>

#include "wave.hh"

namespace arma {

	/// Interpolates function values on irregular grid.
	template <class T>
	class Delaunay_interpolation {

	private:
		typedef T value_type;
		typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
		typedef CGAL::Delaunay_triangulation_2<K> triangulation_type;
		typedef K::Point_2 point_type;
		typedef std::map<point_type, T, K::Less_xy_2> container_type;

	private:
		triangulation_type _points;
		container_type _values;

	public:

		inline void
		insert(T x, T y, T value) {
			const point_type p(x, y);
			this->_points.insert(p);
			this->_values.emplace(p, value);
		}

		T
		operator()(T x, T y);

	};

}

#endif // LINEAR_INTERPOLATON_HH
