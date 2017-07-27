#ifndef OUTPUT_FLAGS_HH
#define OUTPUT_FLAGS_HH

#include <istream>
#include <ostream>
#include <bitset>

namespace arma {

	class Output_flags {
	public:
		typedef std::bitset<32> bitset_type;

		enum Flag: unsigned long {
			None = 0,
			Summary = 1,
			Quantile = 2,
			Waves = 3,
			ACF = 4,
			CSV = 5,
			Blitz = 6,
			Surface = 7
		};

	private:
		bitset_type _flags;

	public:
		Output_flags() = default;

		inline explicit
		Output_flags(Flag f):
		_flags(static_cast<unsigned long>(f)) {}

		inline void
		setf(Flag f) {
			this->_flags.set(size_t(f));
		}

		inline bool
		isset(Flag f) const {
			return this->_flags.test(size_t(f));
		}

		inline bool
		operator==(const Output_flags& rhs) const noexcept {
			return this->_flags == rhs._flags;
		}

		inline bool
		operator!=(const Output_flags& rhs) const noexcept {
			return !operator==(rhs);
		}

		friend std::istream&
		operator>>(std::istream& in, Output_flags& rhs);

		friend std::ostream&
		operator<<(std::ostream& out, const Output_flags& rhs);

	};

	std::istream&
	operator>>(std::istream& in, Output_flags& rhs);

	std::ostream&
	operator<<(std::ostream& out, const Output_flags& rhs);

}

#endif // OUTPUT_FLAGS_HH

