#ifndef VELOCITY_HIGH_AMPLITUDE_REALTIME_SOLVER_HH
#define VELOCITY_HIGH_AMPLITUDE_REALTIME_SOLVER_HH

#include "basic_solver.hh"
#if ARMA_OPENCL
#include "opencl/opencl.hh"
#endif

namespace arma {

	namespace velocity {

		template <class T>
		class High_amplitude_realtime_solver:
		public Velocity_potential_solver<T> {

		#if ARMA_OPENCL
		cl::BufferGL _phi;
		#endif

		public:
			Array4D<T>
			operator()(const Discrete_function<T,3>& zeta) override;
		};

	}

}

#endif // VELOCITY_HIGH_AMPLITUDE_REALTIME_SOLVER_HH
