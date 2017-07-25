#ifndef APMATH_CLOSED_INTERVAL_HH
#define APMATH_CLOSED_INTERVAL_HH

#include <istream>
#include <ostream>
#include <cmath>

namespace arma {

	namespace apmath {

		template <class T>
		class closed_interval {
			T _a, _b;

		public:
			typedef T value_type;

			inline
			closed_interval(T a, T b): _a(a), _b(b) {}
			closed_interval() = default;
			closed_interval(const closed_interval&) = default;
			closed_interval(closed_interval&&) = default;
			~closed_interval() = default;

			inline T
			first() const noexcept {
				return this->_a;
			}

			inline T
			last() const noexcept {
				return this->_b;
			}

			inline bool
			empty() const noexcept {
				return this->_b < this->_a;
			}

			inline bool
			is_point(T eps) const noexcept {
				return std::abs(this->_a - this->_b) < eps;
			}

			inline bool
			valid() const noexcept {
				return this->_a < this->_b;
			}

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const closed_interval<X>& rhs);

			template <class X>
			friend std::istream&
			operator>>(std::istream& in, closed_interval<X>& rhs);

		};

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const closed_interval<T>& rhs);

		template <class T>
		std::istream&
		operator>>(std::istream& in, closed_interval<T>& rhs);

	}

}

#endif // APMATH_CLOSED_INTERVAL_HH
