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

		parameter(T& val): _value(val) {}

		std::istream&
		operator()(std::istream& in, const char* name) {
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

	namespace bits {

		/// ignore lines starting with "#"
		std::istream&
		comment(std::istream& in) {
			char ch = 0;
			while (in >> std::ws >> ch && ch == '#') {
				in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				ch = 0;
			}
			if (ch != 0) { in.putback(ch); }
			return in;
		}

		template<char CH>
		struct const_char {
			friend std::istream&
			operator>>(std::istream& in, const const_char& rhs) {
				char ch;
				if ((ch = in.get()) != CH) {
					in.putback(ch);
					in.setstate(std::ios::failbit);
				}
				return in;
			}
		};
	}

	struct parameter_map {

		typedef std::function<std::istream&(std::istream&, const char*)>
		    read_param;
		typedef std::unordered_map<std::string, read_param> map_type;

		explicit
		parameter_map(map_type&& rhs, bool parens=false):
		_params(rhs),
		_parens(parens)
		{}

		friend std::istream&
		operator>>(std::istream& in, parameter_map& rhs) {
			if (rhs._parens && !(in >> std::ws >> bits::const_char<'{'>())) {
				std::clog << "Expecting \"{\"." << std::endl;
			}
			std::string name;
			while (in >> bits::comment) {
				if (rhs._parens) {
					if (in >> std::ws >> bits::const_char<'}'>()) {
						break;
					}
					in.clear();
				}
				if (!(in >> std::ws && std::getline(in, name, '='))) {
					break;
				}
				trim_right(name);
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
		static void
		trim_right(std::string& rhs) {
			while (!rhs.empty() && rhs.back() <= ' ') { rhs.pop_back(); }
		}

		map_type _params;
		bool _parens;
	};
}

#endif // PARAMS_HH
