#include <gsl/gsl_errno.h>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <unistd.h>

#include "arma_realtime_driver.hh"
#include "velocity/high_amplitude_realtime_solver.hh"
#include "errors.hh"
#include "common_main.hh"

#if ARMA_OPENGL
#include "opengl.hh"
#include <GL/freeglut.h>
#endif

#if ARMA_OPENCL
#include "opencl/opencl.hh"
#endif

arma::ARMA_realtime_driver<ARMA_REAL_TYPE>* driver_ptr = nullptr;

#if ARMA_OPENGL
int dragX = 0;
int dragY = 0;
float scaleX = 1.0f;
float rotateX = 0.0f;
float rotateY = 0.0f;

void
rescale() {
    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
	gluLookAt(
		3, 3, 3,
		0, 0, 0,
		0, 0, 1
	);
	glRotatef(rotateX, 0, 1, 0);
	glRotatef(rotateY, 1, 0, 0);
	glScalef(scaleX, scaleX, scaleX);
}

void
onKeyPressed(unsigned char key, int, int) {
	if (key == 'q') {
		glutLeaveMainLoop();
	}
	if (key == 'r') {
		glutPostRedisplay();
	}
	const float step = 0.2f;
	if (key == ']') {
		if (scaleX >= 1.0f) {
			if (scaleX < 1.0f + step) {
				scaleX = 1.0f;
			}
			scaleX += step;
		} else {
			scaleX *= 2.0f;
		}
		rescale();
		glutPostRedisplay();
	}
	if (key == '[') {
		if (scaleX <= 1.0f) {
			scaleX *= 0.5f;
		} else {
			if (scaleX - step < 1.0f) {
				scaleX = 1.0f;
			} else {
				scaleX -= step;
			}
		}
		rescale();
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
onMouseButton(int, int, int x, int y) {
	dragX = x;
	dragY = y;
}

void
onMouseDrag(int x, int y) {
	rotateX += x - dragX;
	rotateY += y - dragY;
	dragX = x;
	dragY = y;
	rescale();
	glutPostRedisplay();
}

void
onResize(int w, int h) {
	glViewport(0, 0, w, h);
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    gluPerspective( 60, float(w) / h, 0.1, 100 );
	rescale();
}

void
onDisplay() {
	glClearColor(0.25, 0.25, 0.25, 1);
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

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
	glutReshapeFunc(onResize);
	glutMouseFunc(onMouseButton);
	glutMotionFunc(onMouseDrag);
	glEnable(GL_DEPTH_TEST);
}
#else
inline void
init_opengl(int, char**) {}
#endif

template <class T>
void
run_arma(const std::string& input_filename) {
	using namespace arma;
	arma::ARMA_realtime_driver<T> driver;
	driver_ptr = &driver;
	driver.template register_solver<velocity::High_amplitude_realtime_solver<T>>(
		"high_amplitude_realtime"
	);
	register_all_models<T>(driver);
	try {
		driver.open(input_filename);
		driver.generate_wavy_surface();
		driver.compute_velocity_potentials();
	} catch (const PRNG_error& err) {
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
	#if ARMA_PROFILE
	arma_finalise();
	std::exit(0);
	#else
	glutMainLoop();
	#endif
}

