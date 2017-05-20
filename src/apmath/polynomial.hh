#ifndef APMATH_POLYNOMIAL_HH
#define APMATH_POLYNOMIAL_HH

#include <blitz/array.h>
#include <ostream>
#include <initializer_list>

namespace arma {

	namespace apmath {

		/**
		\brief Polynomial class for symbolic computations.
		\date 2017-05-20
		\author Ivan Gankevich

		Coefficient are stored in ascending order from \f$x^0\f$ to \f$x^n\f$.
		*/
		template<class T>
		class Polynomial {

			typedef blitz::Array<T,1> array_type;
			array_type a;

		public:
			inline Polynomial(): a(1) {}
			inline explicit Polynomial(int order): a(order+1) {}
			inline explicit Polynomial(const array_type& coefs): a(coefs) {}
			Polynomial(std::initializer_list<T> coefs);
			inline Polynomial(const Polynomial& rhs) = default;
			inline ~Polynomial() = default;

			Polynomial<T>&
			operator=(const Polynomial& rhs);

			Polynomial
			operator*(const Polynomial& rhs) const;

			inline int
			order() const noexcept {
				return a.extent(0)-1;
			}

			inline size_t
			size() const noexcept {
				return a.size();
			}

			inline const T*
			data() const noexcept {
				return a.data();
			}

			inline T&
			operator[](int i) {
				return a(i);
			}

			inline T
			operator[](int i) const {
				return a(i);
			}

			inline T&
			operator()(int i) {
				return a(i);
			}

			inline T
			operator()(int i) const {
				return a(i);
			}

			template <class X>
			friend std::ostream&
			operator<<(std::ostream& out, const Polynomial<X>& rhs);

		};

		template <class T>
		std::ostream&
		operator<<(std::ostream& out, const Polynomial<T>& rhs);

	}

}

#endif // APMATH_POLYNOMIAL_HH
