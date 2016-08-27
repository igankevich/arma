#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>
#include <sstream>

#include <blitz/array.h>

template <class T>
void
write_matrix(std::ostream& out, blitz::Array<T,2> slice) {
	const int nx = slice.extent(0);
	const int ny = slice.extent(1);
	for (int i=0; i<nx; ++i) {
		for (int j=0; j<ny; ++j) {
			out << slice(i,j) << ' ';
		}
		out << '\n';
	}
}

template <class T>
void
slice_array(const char* filename) {
	using blitz::Array;
	using blitz::Range;
	Array<T, 3> arr;
	std::ifstream in(filename);
	if (!in.is_open()) { throw std::runtime_error("bad file"); }
	if (!(in >> arr)) { throw std::runtime_error("bad array"); }
	const int nslices = arr.extent(0);
	const int ndigits = std::max(1.f, std::ceil(std::log10(float(nslices))));
	for (int i = 0; i < nslices; ++i) {
		Array<T, 2> slice(arr(i, Range::all(), Range::all()));
		std::stringstream slice_filename;
		slice_filename << filename << '-' << std::setfill('0')
		               << std::setw(ndigits) << i;
		std::ofstream out(slice_filename.str());
		write_matrix(out, slice);
	}
}

int
main(int argc, char* argv[]) {
	for (int i = 1; i < argc; ++i) {
		try {
			slice_array<float>(argv[i]);
		} catch (std::exception& err) {
			std::clog << "Error while processing file \"" << argv[i]
			          << "\": " << err.what() << std::endl;
		}
	}
	return 0;
}
