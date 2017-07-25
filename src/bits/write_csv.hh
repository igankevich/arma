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
			const arma::Grid<T,3> outgrid,
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


	}

}

#endif // BITS_WRITE_CSV_HH
