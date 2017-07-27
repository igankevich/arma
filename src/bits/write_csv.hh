#ifndef BITS_WRITE_CSV_HH
#define BITS_WRITE_CSV_HH

#include "grid.hh"
#include <blitz/array.h>
#include <fstream>
#include <string>

namespace arma {

	namespace bits {

		template<class T>
		void
		write_csv(
			std::string filename,
			const blitz::Array<T, 3>& data,
			const Grid<T,3> outgrid,
			const char separator=','
		) {
			std::ofstream out(filename);
			out << 't' << separator
			   << 'x' << separator
			   << 'y' << separator
			   << 'z' << '\n';
			const int nt = data.extent(0);
			const int nx = data.extent(1);
			const int ny = data.extent(2);
			for (int i=0; i<nt; ++i) {
				for (int j=0; j<nx; ++j) {
					for (int k=0; k<ny; ++k) {
						const T x = outgrid(j, 1);
						const T y = outgrid(k, 2);
						out << i << separator
							<< x << separator
							<< y << separator
							<< data(i, j, k)
							<< '\n';
					}
				}
			}
		}

		template<class T>
		void
		write_4d_csv(
			const std::string& filename,
			const blitz::Array<T, 4>& data,
			const Domain2<T>& domain,
			const Grid<T,3> outgrid,
			const char separator=','
		) {
			std::ofstream out(filename);
			out << 't' << separator
			   << 'z' << separator
			   << 'x' << separator
			   << 'y' << separator
			   << "phi" << '\n';
			const int nt = data.extent(0);
			const int nz = data.extent(1);
			const int nx = data.extent(2);
			const int ny = data.extent(3);
			for (int i=0; i<nt; ++i) {
				for (int j=0; j<nz; ++j) {
					const Vec2D<T> p = domain({i,j});
					for (int k=0; k<nx; ++k) {
						for (int l=0; l<ny; ++l) {
							const T x = outgrid(k, 1);
							const T y = outgrid(l, 2);
							out << p(0) << separator
								<< p(1) << separator
								<< x << separator
								<< y << separator
								<< data(i, j, k, l)
								<< '\n';
						}
					}
				}
			}
		}

	}

}

#endif // BITS_WRITE_CSV_HH
