#include "arma_realtime_driver.hh"
#include "velocity/high_amplitude_realtime_solver.hh"

#include <fstream>
#include <algorithm>
#include <iterator>

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
arma::ARMA_realtime_driver<T>::on_display() {
	glBindBuffer(GL_ARRAY_BUFFER, _vbo_phi);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo_phi);
	glColor3f(0.8f, 0, 0);
	glDrawElements(GL_LINE_STRIP, _indices.size(), GL_UNSIGNED_INT, nullptr);
}

template <class T>
void
arma::ARMA_realtime_driver<T>::init_buffers() {
	glGenVertexArrays(1, &_vao_phi);
	glBindVertexArray(_vao_phi);
	using blitz::product;
	const GLsizei num_dimensions = 3;
	GLsizei phi_size = product(this->velocity_potential_grid().num_points())
		* sizeof(T) * num_dimensions;
	glGenBuffers(1, &_vbo_phi);
	glBindBuffer(GL_ARRAY_BUFFER, _vbo_phi);
	glBufferData(GL_ARRAY_BUFFER, phi_size, nullptr, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	using opengl::GL_type;
	glVertexAttribPointer(0, 3, GL_type<T>::value, GL_FALSE, 0, 0);
	typedef velocity::High_amplitude_realtime_solver<T> solver_type;
	solver_type* solver = dynamic_cast<solver_type*>(this->velocity_potential_solver());
	std::clog << "_vbo_phi=" << _vbo_phi << std::endl;
	solver->set_gl_buffer_name(_vbo_phi);
}

template <class T>
void
arma::ARMA_realtime_driver<T>::init_indices() {
	Grid<T,3> grid = this->velocity_potential_grid();
	const int nx = grid.num_points(1);
	const int ny = grid.num_points(2);
	bool dir = true;
	// for explanation see
	// http://andmonahov.blogspot.ru/2012/12/opengl-es-20-4.html
	_indices.clear();
	for (int i=0; i<nx; ++i) {
		if (dir) {
			// left-to-right
			for (int j=0; j<ny; ++j) {
				_indices.emplace_back(i*ny + j);
				_indices.emplace_back((i+1)*ny + j);
			}
			// repeat last index to join the stripes
			_indices.emplace_back((i+1)*ny + (ny-1));
		} else {
			// right-to-left
			for (int j=ny-1; j>=0; --j) {
				_indices.emplace_back(i*ny + j);
				_indices.emplace_back((i+1)*ny + j);
			}
			// repeat last index to join the stripes
			_indices.emplace_back((i+1)*ny);
		}
		dir = !dir;
	}
	glGenBuffers(1, &_ibo_phi);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _ibo_phi);
	glBufferData(
		GL_ELEMENT_ARRAY_BUFFER,
		_indices.size()*sizeof(index_type),
		_indices.data(),
		GL_STATIC_DRAW
	);
	#if ARMA_DEBUG_OPENGL
	{
		std::ofstream out("indices");
		std::copy(
			_indices.begin(),
			_indices.end(),
			std::ostream_iterator<index_type>(out, "\n")
		);
	}
	#endif
}

template <class T>
void
arma::ARMA_realtime_driver<T>::delete_buffers() {
	glDeleteBuffers(1, &_vbo_phi);
}

template <class T>
std::istream&
arma::operator>>(std::istream& in, ARMA_realtime_driver<T>& rhs) {
	in >> static_cast<ARMA_driver<T>&>(rhs);
	rhs.init_buffers();
	rhs.init_indices();
	return in;
}

template class arma::ARMA_realtime_driver<ARMA_REAL_TYPE>;
template std::istream&
arma::operator>>(std::istream& in, ARMA_realtime_driver<ARMA_REAL_TYPE>& rhs);
