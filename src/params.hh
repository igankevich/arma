#ifndef PARAMS_HH
#define PARAMS_HH

#include <functional>
#include <string>
#include <unordered_map>

/// System utilities.
namespace sys {

	/// Configuration file parameter.
	template <class T>
	struct parameter {

		inline
		parameter(T val):
		parameter(val, [](const T,const char*) {})
		{}

		template<class Validate>
		parameter(T val, Validate validate):
		_value(val), _validate(validate)
		{}

		inline std::istream&
		operator()(std::istream& in, const char* name) {
			if (in >> _value) {
				_validate(_value, name);
			}
			return in;
		}

	private:
		T _value;
		std::function<void(const T, const char*)> _validate;
	};

	template <class T>
	parameter<T&>
	make_param(T& val) {
		return parameter<T&>(val);
	}

	template <class T, class Validate>
	parameter<T&>
	make_param(T& val, Validate validate) {
		return parameter<T&>(val, validate);
	}

	template <class T>
	parameter<T>
	wrap_param(T wrapper) {
		return parameter<T>(wrapper);
	}

	/// Configuration file parameters.
	struct parameter_map {

		typedef std::function<std::istream& (std::istream&, const char*)>
		    read_param;
		typedef std::unordered_map<std::string, read_param> map_type;

		inline explicit
		parameter_map(map_type&& rhs, bool parens=false):
		_params(rhs),
		_parens(parens)
		{}

		inline explicit
		parameter_map(map_type&& rhs, std::string name, bool parens=false):
		_params(rhs),
		_name(name),
		_parens(parens)
		{}

		inline void
		insert(const map_type& rhs) {
			this->_params.insert(rhs.begin(), rhs.end());
		}

		friend std::istream&
		operator>>(std::istream& in, parameter_map& rhs);

		friend std::ostream&
		operator<<(std::ostream& out, const parameter_map& rhs);

	private:
		map_type _params;
		std::string _name;
		bool _parens;
	};

	std::istream&
	operator>>(std::istream& in, parameter_map& rhs);

	std::ostream&
	operator<<(std::ostream& out, const parameter_map& rhs);

}

#endif // PARAMS_HH
