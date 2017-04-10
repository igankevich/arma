#ifndef LINEAR_INTERPOLATON_HH
#define LINEAR_INTERPOLATON_HH

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Interpolation_traits_2.h>
#include <CGAL/natural_neighbor_coordinates_2.h>
#include <CGAL/interpolation_functions.h>

#include <vector>
#include <map>
#include "wave.hh"

namespace arma {

	typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
	typedef CGAL::Delaunay_triangulation_2<K>             Delaunay_triangulation;
	typedef K::FT                                         Coord_type;
	typedef K::Point_2                                    Point;
	typedef std::map<Point, Coord_type, K::Less_xy_2>     Function_values;
	typedef CGAL::Data_access<Function_values>            Value_access;

	template<class T>
	Delaunay_triangulation
	create_triangulation(const std::vector< Wave<T> >& waves);

	template<class T>
	Function_values
	create_function_values(const std::vector< Wave<T> >& waves);

	template<class T>
	T interpolate(
		Point p,
		const Delaunay_triangulation& triangulation,
		const Function_values& function_values
	);

	template<class T>
	T
	interpolate(Point p, const std::vector< Wave<T> >& waves);

}

#endif // LINEAR_INTERPOLATON_HH
