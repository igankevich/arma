#include <gsl/gsl_errno.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <unistd.h>

#include "arma_realtime_driver.hh"
#include "velocity/high_amplitude_realtime_solver.hh"

#if ARMA_OPENGL
#include "opengl.hh"
#include <GL/freeglut.h>
#endif

#if ARMA_OPENCL
#include "opencl/opencl.hh"
#endif

void
print_exception_and_terminate() {
	if (std::exception_ptr ptr = std::current_exception()) {
		try {
			std::rethrow_exception(ptr);
		#if ARMA_OPENCL
		} catch (cl::Error err) {
			std::cerr << err << std::endl;
			std::abort();
		#endif
		} catch (const std::exception& e) {
			std::cerr << "ERROR: " << e.what() << std::endl;
		} catch (...) { std::cerr << "UNKNOWN ERROR. Aborting." << std::endl; }
	}
	std::exit(1);
}

void
print_error_and_continue(
	const char* reason,
	const char* file,
	int line,
	int gsl_errno
) {
	std::cerr << "GSL error reason: " << reason << '.' << std::endl;
}

void
usage(char* argv0) {
	std::cout
		<< "USAGE: "
		<< (argv0 == nullptr ? "arma" : argv0)
		<< " -c CONFIGFILE\n";
}

template<class Solver, class Driver>
void
register_vpsolver(Driver& drv, std::string key) {
	drv.template register_velocity_potential_solver<Solver>(key);
}

void
init_opencl() {
	::arma::opencl::init();
}

arma::ARMA_realtime_driver<ARMA_REAL_TYPE>* driver_ptr = nullptr;

#if ARMA_OPENGL
void
onKeyPressed(unsigned char key, int, int) {
	if (key == 'q') {
		glutLeaveMainLoop();
	}
	if (key == 'r') {
		glutPostRedisplay();
	}
}

float ver[8][3] =
{
    {-1.0,-1.0,1.0},
    {-1.0,1.0,1.0},
    {1.0,1.0,1.0},
    {1.0,-1.0,1.0},
    {-1.0,-1.0,-1.0},
    {-1.0,1.0,-1.0},
    {1.0,1.0,-1.0},
    {1.0,-1.0,-1.0},
};

GLfloat color[8][3] =
{
    {0.0,0.0,0.0},
    {1.0,0.0,0.0},
    {1.0,1.0,0.0},
    {0.0,1.0,0.0},
    {0.0,0.0,1.0},
    {1.0,0.0,1.0},
    {1.0,1.0,1.0},
    {0.0,1.0,1.0},
};

void quad(int a,int b,int c,int d)
{
    glBegin(GL_QUADS);
    glColor3fv(color[a]);
    glVertex3fv(ver[a]);

    glColor3fv(color[b]);
    glVertex3fv(ver[b]);

    glColor3fv(color[c]);
    glVertex3fv(ver[c]);

    glColor3fv(color[d]);
    glVertex3fv(ver[d]);
    glEnd();
}

void colorcube()
{
    quad(0,3,2,1);
    quad(2,3,7,6);
    quad(0,4,7,3);
    quad(1,2,6,5);
    quad(4,5,6,7);
    quad(0,1,5,4);
}


void
onDisplay() {
	std::clog << __func__ << std::endl;
	glClearColor(0.25, 0.25, 0.25, 1);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    int w = glutGet( GLUT_WINDOW_WIDTH );
    int h = glutGet( GLUT_WINDOW_HEIGHT );
    gluPerspective( 60, float(w) / h, 0.1, 100 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
	gluLookAt(
		3, 3, 3,
		0, 0, 0,
		0, 0, 1
	);

	glRotatef( 0, 1.0, 0.0, 0.0 );
	glRotatef( 0, 0.0, 1.0, 0.0 );

	if (driver_ptr) {
		driver_ptr->on_display();
	}

	glutSwapBuffers();
}

void
init_opengl(int argc, char* argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
	int wnd_w = 800;
	int wnd_h = 600;
	int screen_w = glutGet(GLUT_SCREEN_WIDTH);
	int screen_h = glutGet(GLUT_SCREEN_HEIGHT);
	glutInitWindowSize(wnd_w, wnd_h);
	glutInitWindowPosition((screen_w - wnd_w) / 2, (screen_h - wnd_h) / 2);
	glutCreateWindow("arma-realtime");
	glutDisplayFunc(onDisplay);
	glutKeyboardFunc(onKeyPressed);
	glEnable(GL_DEPTH_TEST);
}
#else
inline void
init_opengl(int, char**) {}
#endif

int
main(int argc, char* argv[]) {

	/// Print GSL errors and proceed execution.
	/// Throw domain-specific exception later.
	gsl_set_error_handler(print_error_and_continue);
	std::set_terminate(print_exception_and_terminate);

	init_opengl(argc, argv);
	init_opencl();

	using namespace arma;

	/// floating point type (float, double, long double or multiprecision number
	/// C++ class)
	typedef ARMA_REAL_TYPE T;

	std::string input_filename;
	bool help_requested = false;
	int opt = 0;
	while ((opt = ::getopt(argc, argv, "c:h")) != -1) {
		switch (opt) {
			case 'c':
				input_filename = ::optarg;
				break;
			case 'h':
				help_requested = true;
				break;
		}
	}

	arma::ARMA_realtime_driver<T> driver;
	driver_ptr = &driver;
	if (help_requested || input_filename.empty()) {
		usage(argv[0]);
	} else {
		using namespace velocity;
		register_vpsolver<High_amplitude_realtime_solver<T>>(
			driver,
			"high_amplitude_realtime"
		);
		std::ifstream cfg(input_filename);
		if (!cfg.is_open()) {
			std::clog << "Cannot open input file "
				"\"" << input_filename << "\"."
				<< std::endl;
			throw std::runtime_error("bad input file");
		}
		write_key_value(std::clog, "Input file", input_filename);
		cfg >> driver;
		try {
			driver.generate_wavy_surface();
			driver.compute_velocity_potentials();
		} catch (const prng_error& err) {
			if (err.ngenerators() == 0) {
				std::cerr << "No parallel Mersenne Twisters configuration is found. "
					"Please, generate sufficient number of MTs with dcmt programme."
					<< std::endl;
			} else {
				std::cerr << "Insufficient number of parallel Mersenne Twisters found. "
					"Please, generate at least " << err.nparts() << " MTs for this run."
					<< std::endl;
			}
		}
	}
	glutMainLoop();
	return 0;
}

