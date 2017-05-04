#ifndef ARMA_REALTIME_DRIVER_HH
#define ARMA_REALTIME_DRIVER_HH

#include <istream>

#include "arma_driver.hh"
#include "opengl.hh"

namespace arma {

	template <class T>
	class ARMA_realtime_driver: public ARMA_driver<T> {

		GLuint _glphi;

	public:
		ARMA_realtime_driver();
		~ARMA_realtime_driver();

	private:
		void
		init_buffers();

		void
		delete_buffers();

		template<class X>
		friend std::istream&
		operator>>(std::istream& in, ARMA_realtime_driver<X>& rhs);

	};

	template <class T>
	std::istream&
	operator>>(std::istream& in, ARMA_realtime_driver<T>& rhs);

}

#endif // ARMA_REALTIME_DRIVER_HH
