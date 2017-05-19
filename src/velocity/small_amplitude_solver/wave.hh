#ifndef WAVE_HH
#define WAVE_HH

#include <ostream>

namespace arma {

	namespace bits {

		template<class T>
		class Wave {
			int i = 0, j = 0;
			T k = 0;
			T height = 0;
			T period = 0;

		public:
			inline
			Wave(int i_, int j_, T k_, T h, T p):
			i(i_), j(j_), k(k_), height(h), period(p)
			{}

			Wave() = default;

			inline T wave_number() const noexcept { return k; }
			inline T amplitude_rating() const noexcept { return height / period; }
			inline int x() const noexcept { return i; }
			inline int y() const noexcept { return j; }

			template<class X>
			friend std::ostream&
			operator<<(std::ostream& out, const Wave<X>& rhs);

		};

		template<class T>
		std::ostream&
		operator<<(std::ostream& out, const Wave<T>& rhs);

	}

}

#endif // WAVE_HH
