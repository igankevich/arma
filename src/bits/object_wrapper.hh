#ifndef BITS_OBJECT_WRAPPER_HH
#define BITS_OBJECT_WRAPPER_HH

#include <iostream>
#include <string>
#include <functional>
#include <unordered_map>

namespace arma {

	namespace bits {

		/// Helper class that constructs class instances by name.
		template <class T>
		class Object_wrapper {
			typedef T solver_type;
			typedef std::string key_type;
			typedef std::function<solver_type*()> value_type;
			typedef std::unordered_map<std::string, value_type> map_type;
			solver_type*& _solver;
			const map_type& _constructors;
			key_type _name;

		public:
			explicit
			Object_wrapper(solver_type*& solver, const map_type& constructors):
			_solver(solver),
			_constructors(constructors)
			{}

			friend std::istream&
			operator>>(std::istream& in, Object_wrapper& rhs) {
				std::string name;
				in >> std::ws >> name;
				auto result = rhs._constructors.find(name);
				if (result == rhs._constructors.end()) {
					in.setstate(std::ios::failbit);
					std::cerr << "Invalid object: " << name << std::endl;
					throw std::runtime_error("bad object");
				} else {
					rhs._name = result->first;
					rhs._solver = result->second();
					in >> *rhs._solver;
				}
				return in;
			}

			const key_type&
			name() const noexcept {
				return this->_name;
			}

		};

	}

}

#endif // BITS_OBJECT_WRAPPER_HH
