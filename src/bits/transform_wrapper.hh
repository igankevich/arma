#ifndef BITS_TRANSFORM_WRAPPER_HH
#define BITS_TRANSFORM_WRAPPER_HH

#include <string>
#include <iostream>
#include <stdexcept>

namespace arma {

	namespace bits {

		template <class Tr>
		class Transform_wrapper {
			Tr& _transform;
			bool& _linear;

		public:
			inline explicit
			Transform_wrapper(Tr& tr, bool& linear):
			_transform(tr),
			_linear(linear)
			{}

			friend std::istream&
			operator>>(std::istream& in, Transform_wrapper& rhs) {
				std::string name;
				in >> std::ws >> name;
				if (name == "nit") {
					in >> rhs._transform;
					rhs._linear = false;
				} else if (name == "none") {
					rhs._linear = true;
				} else {
					in.setstate(std::ios::failbit);
					std::cerr << "Invalid transform: " << name << std::endl;
					throw std::runtime_error("bad transform");
				}
				return in;
			}

		};

	}

}

#endif // BITS_TRANSFORM_WRAPPER_HH
