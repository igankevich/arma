#include "arma_realtime_driver.hh"
#include "velocity/high_amplitude_realtime_solver.hh"

template <class T>
arma::ARMA_realtime_driver<T>::ARMA_realtime_driver():
ARMA_driver<T>()
{}

template <class T>
arma::ARMA_realtime_driver<T>::~ARMA_realtime_driver() {
	delete_buffers();
}

template <class T>
void
arma::ARMA_realtime_driver<T>::init_buffers() {
	glGenBuffers(1, &_glphi);
	using blitz::product;
	const GLsizei num_dimensions = 3;
	GLsizei phi_size = product(this->velocity_potential_grid().num_points())
		* sizeof(T) * num_dimensions;
	glBindBuffer(GL_ARRAY_BUFFER, _glphi);
	glBufferData(GL_ARRAY_BUFFER, phi_size, nullptr, GL_DYNAMIC_DRAW);
	typedef velocity::High_amplitude_realtime_solver<T> solver_type;
	solver_type* solver = dynamic_cast<solver_type*>(this->velocity_potential_solver());
	std::clog << "_glphi=" << _glphi << std::endl;
	solver->set_gl_buffer_name(_glphi);
}

template <class T>
void
arma::ARMA_realtime_driver<T>::delete_buffers() {
	glDeleteBuffers(1, &_glphi);
}

template <class T>
std::istream&
arma::operator>>(std::istream& in, ARMA_realtime_driver<T>& rhs) {
	in >> static_cast<ARMA_driver<T>&>(rhs);
	rhs.init_buffers();
	return in;
}

template class arma::ARMA_realtime_driver<ARMA_REAL_TYPE>;
template std::istream&
arma::operator>>(std::istream& in, ARMA_realtime_driver<ARMA_REAL_TYPE>& rhs);
