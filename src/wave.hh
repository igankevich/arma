#ifndef WAVE_HH
#define WAVE_HH

#include <ostream>

namespace arma {

	/**
	   \brief A wave with height, period, wave number, \f$X\f$ and \f$Y\f$
	   coordinate.
	 */
	template<class T>
	class Wave {

	private:
		int _i = 0, _j = 0;
		T _wavenum = 0;
		T _height = 0;
		T _period = 0;

	public:

		Wave() = default;

		inline
		Wave(int i, int j, T k, T h, T p) noexcept:
		_i(i), _j(j), _wavenum(k), _height(h), _period(p)
		{}

		inline T
		wave_number() const noexcept {
			return this->_wavenum;
		}

		inline T
		amplitude_rating() const noexcept {
			return this->_height / this->_period;
		}

		inline int
		x() const noexcept {
			return this->_i;
		}

		inline int
		y() const noexcept {
			return this->_j;
		}

		template<class X>
		friend std::ostream&
		operator<<(std::ostream& out, const Wave<X>& rhs);

	};

	template<class T>
	std::ostream&
	operator<<(std::ostream& out, const Wave<T>& rhs);

}

#endif // WAVE_HH
