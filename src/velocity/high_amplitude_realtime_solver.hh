#ifndef VELOCITY_HIGH_AMPLITUDE_REALTIME_SOLVER_HH
#define VELOCITY_HIGH_AMPLITUDE_REALTIME_SOLVER_HH

#include "basic_solver.hh"
#include "grid.hh"
#if ARMA_OPENCL
#include "opencl/opencl.hh"
#endif

namespace arma {

	namespace velocity {

		template <class T>
		class High_amplitude_realtime_solver:
		public Velocity_potential_solver<T> {

		#if ARMA_OPENCL
			#if ARMA_OPENGL
				typedef cl::BufferGL buffer_type;
			#else
				typedef cl::Buffer buffer_type;
			#endif
		#else
			typedef cl::Buffer buffer_type;
		#endif

		buffer_type _phi;
		cl::Buffer _wfunc;
		cl::Buffer _sfunc;

		public:
			Array4D<T>
			operator()(const Discrete_function<T,3>& zeta) override;

		private:
			void
			compute_window_function(const Grid<T,3>& domain);

			void
			compute_second_function(const Discrete_function<T,3>& zeta);

			void
			compute_velocity_field(const Grid<T,3>& domain);
		};

	}

}

#endif // VELOCITY_HIGH_AMPLITUDE_REALTIME_SOLVER_HH
