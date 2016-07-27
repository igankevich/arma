#ifndef PARAMS_HH
#define PARAMS_HH

#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <string>
#include <limits>

/// Various system utilities.
namespace sys {

	template <class T>
	struct parameter {

		parameter(T& val) : _value(val) {}

		std::istream& operator()(std::istream& in, const char* name) {
			return in >> _value;
		}

	private:
		T& _value;
	};

	template <class T>
	parameter<T>
	make_param(T& val) {
		return parameter<T>(val);
	}

	struct parameter_map {

		typedef std::function<std::istream&(std::istream&, const char*)>
		    read_param;
		typedef std::unordered_map<std::string, read_param> map_type;

		explicit parameter_map(map_type&& rhs) : _params(rhs) {}

		friend std::istream& operator>>(std::istream& in, parameter_map& rhs) {
			std::string name;
			while (in >> std::ws && std::getline(in, name, '=')) {
				/// ignore lines starting with "#"
				if (name.size() > 0 && name[0] == '#') {
					in.ignore(std::numeric_limits<std::streamsize>::max(),
					          '\n');
				}
				auto result = rhs._params.find(name);
				if (result == rhs._params.end()) {
					std::clog << "Unknown parameter: " << name << std::endl;
					in.setstate(std::ios::failbit);
				} else {
					result->second(in, name.data());
				}
			}
			return in;
		}

	private:
		map_type _params;
	};
}

#endif // PARAMS_HH
