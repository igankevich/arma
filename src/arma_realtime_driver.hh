#ifndef ARMA_REALTIME_DRIVER_HH
#define ARMA_REALTIME_DRIVER_HH

#include <istream>
#include <vector>

#include "arma_driver.hh"
#include "opengl.hh"

namespace arma {

	template <class T>
	class ARMA_realtime_driver: public ARMA_driver<T> {

		typedef GLuint index_type;

		GLuint _vbo_phi;
		GLuint _ibo_phi;
		/// Indices of a vertex array for drawing wavy surface
		/// with triangle strips.
		std::vector<index_type> _indices;

	public:
		ARMA_realtime_driver();
		virtual ~ARMA_realtime_driver();

		void
		on_display();

	private:
		void
		init_buffers();

		void
		delete_buffers();

		void
		init_indices();

		template<class X>
		friend std::istream&
		operator>>(std::istream& in, ARMA_realtime_driver<X>& rhs);

	};

	template <class T>
	std::istream&
	operator>>(std::istream& in, ARMA_realtime_driver<T>& rhs);

}

#endif // ARMA_REALTIME_DRIVER_HH
