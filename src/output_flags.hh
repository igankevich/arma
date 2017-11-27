#ifndef OUTPUT_FLAGS_HH
#define OUTPUT_FLAGS_HH

#include <bitset>
#include <istream>
#include <ostream>
#include <string>
#include <type_traits>

#if ARMA_BSCHEDULER
#include <bscheduler/api.hh>
#endif

namespace arma {

	/// Flags that determine what and how is written to disk.
	class Output_flags {

	public:
		typedef std::bitset<32> bitset_type;

		enum Flag : unsigned long {
			None = 0,
			Summary = 1,
			Quantile = 2,
			Waves = 3,
			ACF = 4,
			CSV = 5,
			Blitz = 6,
			Binary = 7,
			Surface = 8
		};

	private:
		bitset_type _flags;

	public:
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

		#if ARMA_BSCHEDULER
		void
		write(sys::pstream& out) const;

		void
		read(sys::pstream& in);

		#endif

	private:
		/// Infer default values when no output format is specified.
		void
		prune();

	};

	std::istream&
	operator>>(std::istream& in, Output_flags& rhs);

	std::ostream&
	operator<<(std::ostream& out, const Output_flags& rhs);

	std::string
	get_filename(const std::string& prefix, Output_flags::Flag flag);

	inline std::string
	get_surface_filename(Output_flags::Flag flag) {
		return get_filename("zeta", flag);
	}

	inline std::string
	get_velocity_filename(Output_flags::Flag flag) {
		return get_filename("phi", flag);
	}

	#if ARMA_BSCHEDULER
	inline sys::pstream&
	operator<<(sys::pstream& out, const Output_flags& rhs) {
		rhs.write(out);
		return out;
	}

	inline sys::pstream&
	operator>>(sys::pstream& in, Output_flags& rhs) {
		rhs.read(in);
		return in;
	}
	#endif

}

#endif // OUTPUT_FLAGS_HH
