#ifndef UTILITY_CONST_CHAR_HH
#define UTILITY_CONST_CHAR_HH

#include <istream>

namespace arma {

	namespace util {

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

}

#endif // UTILITY_CONST_CHAR_HH
