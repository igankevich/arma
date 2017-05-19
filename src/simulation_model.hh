#ifndef SIMULATION_MODEL_HH
#define SIMULATION_MODEL_HH

#include <istream>
#include <ostream>

namespace arma {

	enum struct Simulation_model {
		Autoregressive,
		Moving_average,
		ARMA,
		Longuet_Higgins,
		Plain_wave_model
	};

	std::istream&
	operator>>(std::istream& in, Simulation_model& rhs);

	const char*
	to_string(Simulation_model rhs);

	std::ostream&
	operator<<(std::ostream& out, const Simulation_model& rhs);
}

#endif // SIMULATION_MODEL_HH
