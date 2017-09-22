#ifndef VELOCITY_HIGH_AMPLITUDE_REALTIME_SOLVER_HH
#define VELOCITY_HIGH_AMPLITUDE_REALTIME_SOLVER_HH

#include "basic_solver.hh"
#include "grid.hh"

#if ARMA_OPENCL
#include <clFFT.h>
#include "opencl/opencl.hh"
#endif

#if ARMA_OPENGL
#include "opengl.hh"
#endif

namespace arma {

	namespace velocity {

		/**
		\brief A version of \link High_amplitude_solver \endlink
		with shared OpenCL/OpenGL buffers for real-time rendering.

		- OpenCL/OpenGL sharing is used only when it is supported by the current
		  OpenCL runtime.
		- Only the first OpenCL device is used for computations. Multiple
		  devices are not supported yet. The name of the device can be specified
		  in `opencl.conf` the configuration file.

		\see \link High_amplitude_solver \endlink
		\see \link opencl/opencl.hh \endlink
		\ingroup solvers
		*/
		template <class T>
		class High_amplitude_realtime_solver:
		public Velocity_potential_solver<T> {

			/// Velocity potential scala field.
			cl::Buffer _phi;
			cl::Buffer _wfunc;

			/// Velocity potential vector field (CL/GL shared buffer).
			cl::Buffer _vphi;
			/// GL buffer name.
			GLuint _glphi = 0;

			clfftSetupData _fft;
			clfftPlanHandle _fftplan;

		public:
			High_amplitude_realtime_solver();
			~High_amplitude_realtime_solver();

			Array4D<T>
			operator()(const Discrete_function<T,3>& zeta) override;

			#if ARMA_OPENGL
			inline void
			set_gl_buffer_name(GLuint name) noexcept {
				this->_glphi = name;
			}
			#endif

		private:
			void
			setup(
				const Discrete_function<T,3>& zeta,
				const Grid<T,3>& grid,
				const Grid<T,3>& wngrid
			);

			void
			compute_window_function(const Grid<T,3>& domain);

			void
			interpolate_window_function(const Grid<T,3>& domain);

			void
			compute_second_function(
				const Discrete_function<T,3>& zeta,
				const int idx_t
			);

			void
			multiply_functions(const Grid<T,3>& grid);

			void
			fft(const Grid<T,3>& grid, clfftDirection dir);

			void
			create_vector_field(const Grid<T,3>& grid);
		};

	}

}

#endif // VELOCITY_HIGH_AMPLITUDE_REALTIME_SOLVER_HH
