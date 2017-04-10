#include "linear_interpolation.hh"

template<class T>
arma::Delaunay_triangulation
arma::create_triangulation(const std::vector< Wave<T> >& waves) {
	Delaunay_triangulation triangulation;
	int n = waves.size();
	for (int i=0; i<n; ++i) {
		triangulation.insert(Point(waves[i].x(), waves[i].y()));
	}
	return triangulation;
}

template<class T>
arma::Function_values
arma::create_function_values(const std::vector< Wave<T> >& waves) {
	Function_values values;
	int n = waves.size();
	for (int i=0; i<n; ++i) {
		Point p(waves[i].x(), waves[i].y());
		values.emplace(p, waves[i].wave_number());
	}
	return values;
}

template<class T>
T
arma::interpolate(
	Point p,
	const Delaunay_triangulation& triangulation,
	const Function_values& function_values
) {
	std::vector< std::pair<Point, Coord_type> > coords;
	Coord_type norm =
		CGAL::natural_neighbor_coordinates_2
		(triangulation, p,std::back_inserter(coords)).second;

	Coord_type res =  CGAL::linear_interpolation(coords.begin(), coords.end(),
					  norm,
					  Value_access(function_values));

	return res;
}

template<class T>
T
arma::interpolate(Point p, const std::vector< Wave<T> >& waves) {
//	Delaunay_triangulation triangulation = create_triangulation(waves);
//	Function_values function_values = create_function_values(waves);
	Delaunay_triangulation triangulation;
	int n = waves.size();
	for (int i=0; i<n; ++i) {
		triangulation.insert(Point(waves[i].x(), waves[i].y()));
	}
	Function_values function_values;
	for (int i=0; i<n; ++i) {
		Point p(waves[i].x(), waves[i].y());
		function_values.emplace(p, waves[i].wave_number());
	}
	std::vector< std::pair<Point, Coord_type> > coords;
	Coord_type norm =
		CGAL::natural_neighbor_coordinates_2
		(triangulation, p,std::back_inserter(coords)).second;

	Coord_type res =  CGAL::linear_interpolation(coords.begin(), coords.end(),
					  norm,
					  Value_access(function_values));

	return res;
}

template
arma::Delaunay_triangulation
arma::create_triangulation(const std::vector< Wave<ARMA_REAL_TYPE> >& waves);

template
arma::Function_values
arma::create_function_values(const std::vector< Wave<ARMA_REAL_TYPE> >& waves);

template
ARMA_REAL_TYPE
arma::interpolate(
	Point p,
	const Delaunay_triangulation& triangulation,
	const Function_values& function_values
);

template
ARMA_REAL_TYPE
arma::interpolate(Point p, const std::vector< Wave<ARMA_REAL_TYPE> >& waves);
