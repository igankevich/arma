#include <algorithm>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "arma_driver.hh"
#include "register_all.hh"
#include "opengl.hh"
#include "types.hh"

#include <SDL.h>
#include <imgui/imgui.h>
#include <imgui/imgui_sdl.h>

#define SDL_CHECK(ret) \
	if (ret) { \
		throw ::std::runtime_error(SDL_GetError()); \
	}

typedef ARMA_REAL_TYPE T;

const int fps = 60;

SDL_Window* window = nullptr;
SDL_GLContext glcontext;
std::atomic<bool> running(false);
std::thread arma_thread;
char arma_config[4096*4] = R"(
#model = AR {
#	out_grid = (100,40,40)
#	acf = {
#		func = standing_wave
#		grid = (10,10,10) : (2.5,5,5)
#	}
#	order = (20,20,20)
#	output = surface
#}

model = MA {
	out_grid = (200,40,40)
	acf = {
		func = propagating_wave
		grid = (20,10,10) : (10,5,5)
	}
	order = (20,10,10)
	algorithm = fixed_point_iteration
	output = none
	validate = 0
}

velocity_potential_solver = linear {
#	wnmax = from (0,0) to (0,0.25) npoints (2,2)
	depth = 12
	domain = from (10,-12) to (10,3) npoints (1,128)
}
)";

using namespace arma;

typedef ARMA_REAL_TYPE Real;

enum Projection {
	PROJECTION_X = 0,
	PROJECTION_Y = 1,
	PROJECTION_T = 2,
	PROJECTION_NONE = -1
};

Array3D<Real> func;
Vector<Real, 3> delta(0.1, 1.0, 1.0);
Vector<int, 3> dimensions(blitz::firstDim, blitz::secondDim, blitz::thirdDim);
Vector<char, 3> dimension_names('t', 'x', 'y');


// ACF parameters
const char* acf_names[] = {"standing_wave", "propagating_wave"};
int current_acf_index = 0;
float acf_amplitude = 1;
float acf_velocity = 1;
Vector<float,3> acf_alpha(0.24f,0.06f,0.06f);
Vector<float,2> acf_beta(0.8f,0.0f);
Vector<int,3> acf_size(20,20,20);

const Real ROT_STEP = 90;

Projection proj = PROJECTION_NONE;
bool paused = true;
int timer = 0;
int dimension_order = 0; /// dimension dimension_order
int part_count = 4;
Real interval_factor = 0;
int tail = 0;

void
generate_configuration(std::stringstream& result) {
	const bool standing =
		acf_names[current_acf_index] == std::string("standing_wave");
	if (standing) {
		result << "model = AR {\n";
		result << "algorithm = choi_recursive\n";
	} else {
		result << "model = MA {\n";
		result << "algorithm = fixed_point_iteration\n";
		result << "validate = 0\n";
	}
	result << "acf = {\n";
	if (standing) {
		result << "grid = " << acf_size << " : (5,10,10)\n";
	} else {
		result << "grid = " << acf_size << " : (10,5,5)\n";
	}
	result << "amplitude = " << acf_amplitude << "\n";
	result << "velocity = " << acf_velocity << "\n";
	result << "alpha = " << acf_alpha << "\n";
	result << "beta = " << acf_beta << "\n";
	result << "func = " << acf_names[current_acf_index] << "\n";
	result << "}\n";
	result << "order = " << acf_size << "\n";
	result << "out_grid = (100,40,40)\n";
	result << "output = none\n";
	result << "}\n";
	result << R"(
velocity_potential_solver = high_amplitude {
	depth = 12
	domain = from (10,-12) to (10,3) npoints (1,128)
}
)";
}

void
draw_vertex(const Shape3D& c, const Shape3D& offset, float alpha) {
	glColor4f(0.85, 0.85, 0.85, alpha);
	glVertex3f(c[1] + offset[1], c[2] + offset[2], func(c));
}

void
drawSeries(size_t t, float alpha) {
	const Shape3D& size = func.shape();
	const Shape3D offset = -size / 2;
	int x1 = size[1];
	int y1 = size[2];
	for (int i = 0; i < x1; i++) {
		glBegin(GL_LINE_STRIP);
		for (int j = 0; j < y1; j++) {
			draw_vertex(Shape3D(t, i, j), offset, alpha);
		}
		glEnd();
	}
	for (int j = 0; j < y1; j++) {
		glBegin(GL_LINE_STRIP);
		for (int i = 0; i < x1; i++) {
			draw_vertex(Shape3D(t, i, j), offset, alpha);
		}
		glEnd();
	}
}

void
draw_axis(const GLfloat v[3]) {
	glColor3fv(v);
	glBegin(GL_LINES);
	glVertex3f(0, 0, 0);
	glVertex3fv(v);
	glEnd();
}

void
resetView() {
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	const Real m = std::min(delta[1], delta[2]);
	glScalef(delta[1] / m, delta[2] / m, 1.0f);
	// float3 offset = (dom.max() - dom.min())*-0.5;
	//	glTranslatef(offset[1], offset[2], -100.0f);
	glTranslatef(0, 0, -100.0f);
	glRotatef(-30.0f, 1, 0, 0);
}

template<class T, int N>
void
rotate(blitz::TinyVector<T, N>& rhs) {
	std::rotate(rhs.begin(), rhs.end()-1, rhs.end());
}

void
rotate_dimensions() {
	rotate(delta);
	rotate(dimensions);
	rotate(dimension_names);
	func.transposeSelf(dimensions(0), dimensions(1), dimensions(2));
	timer = std::min(timer, func.extent(0));
	resetView();
	std::clog << "Dimensions = " << dimension_names << std::endl;
}

void
onDisplay() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0.25, 0.25, 0.25, 1.0);

	const int tl = std::min(tail, timer);
	for (int t = timer - tl, i = 1; t <= timer; ++t, ++i) {
		drawSeries(t, i / (tl + 1.0));
	}

	std::stringstream str;
	str << "t=" << timer << '/' << func.extent(0) - 1;

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_DEPTH_TEST);

	glRasterPos2i(-1, -1);
	glColor3f(1, 1, 1);
	//glutBitmapString(GLUT_BITMAP_HELVETICA_18,
	//                 (const unsigned char*)str.str().c_str());

	glEnable(GL_DEPTH_TEST);
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	ImGui_SDL_NewFrame(window);
	const int line_height = ImGui::GetTextLineHeight();
	ImGui::SetNextWindowSize(ImVec2(400, line_height*50), ImGuiSetCond_FirstUseEver);
	ImGui::Begin("ARMA configuration");
	if (ImGui::Button(running ? "Please, wait..." : "Compute")) {
		if (!running) {
			if (arma_thread.joinable()) {
				arma_thread.join();
			}
			running = true;
			arma_thread = std::thread([&] () {
				try {
					ARMA_driver<T> driver;
					register_all_models<T>(driver);
					register_all_solvers<T>(driver);
					std::stringstream cfg;
					generate_configuration(cfg);
					cfg >> driver;
					driver.generate_wavy_surface();
					func.reference(driver.wavy_surface());
				} catch (const std::exception& err) {
					std::cerr << "Error: " << err.what() << std::endl;
				}
				running = false;
			});
		}
	}

	ImGui::Combo("ACF type", &current_acf_index, acf_names, 2);
	ImGui::SliderFloat("amplitude", &acf_amplitude, 0.1f, 10.0f, "%.2f");
	ImGui::SliderFloat("velocity", &acf_velocity, -10.0f, 10.0f, "%.2f");
	ImGui::SliderFloat("alpha_t", &acf_alpha(0), 0.0f, 10.0f, "%.2f");
	ImGui::SliderFloat("alpha_x", &acf_alpha(1), 0.0f, 10.0f, "%.2f");
	ImGui::SliderFloat("alpha_y", &acf_alpha(2), 0.0f, 10.0f, "%.2f");
	ImGui::SliderFloat("kx", &acf_beta(0), -10.0f, 10.0f, "%.2f");
	ImGui::SliderFloat("ky", &acf_beta(1), -10.0f, 10.0f, "%.2f");
//	ImGui::InputTextMultiline(
//		"##source",
//		arma_config,
//		sizeof(arma_config),
//		ImVec2(-1.0f, line_height*40),
//		ImGuiInputTextFlags_AllowTabInput
//	);
	ImGui::End();
	ImGui::Render();

	SDL_GL_SwapWindow(window);
}

void
onMouseDrag(int x, int y) {
	glRotatef(x, 0, 1, 0);
	glRotatef(y, 1, 0, 0);
}

void
onResize(int w, int h) {
	glViewport(0, 0, w, h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	// gluPerspective(30.0, (GLfloat)w / (GLfloat)h, 0.1, 100000.0f);
	// glOrtho(-10, 10, -10, 10, 0.1, 1000);
	// The following code is a fancy bit of math that is eqivilant to calling:
	// // gluPerspective( fieldOfView/2.0f, width/height , 0.1f, 255.0f )
	// // We do it this way simply to avoid requiring glu.h
	GLfloat zNear = 0.1f;
	GLfloat zFar = 25500.0f;
	GLfloat aspect = float(w) / float(h);
	GLfloat fH = tan(float(60 / 360.0f * 3.14159f)) * zNear;
	GLfloat fW = fH * aspect;
	glFrustum(-fW, fW, -fH, fH, zNear, zFar);
	glMatrixMode(GL_MODELVIEW);
	//	resetView();
}

void
onKeyPressed(SDL_Keycode key, Uint16 mods) {
	if (key == 'q') exit(0);
	if (key == 32) paused = !paused;

	if (key == 'a') glRotatef(ROT_STEP, 1, 0, 0);
	if (key == 's') glRotatef(ROT_STEP, 0, 1, 0);
	if (key == 'd') glRotatef(ROT_STEP, 0, 0, 1);

	if (key == 'z') glRotatef(-ROT_STEP, 1, 0, 0);
	if (key == 'x') glRotatef(-ROT_STEP, 0, 1, 0);
	if (key == 'c') glRotatef(-ROT_STEP, 0, 0, 1);

	if (key == ']') glScalef(1.5f, 1.5f, 1.5f);
	if (key == '[') glScalef(0.9f, 0.9f, 0.9f);

	if (key == 'h') glTranslatef(2.0f, 0.0f, 0.0f);
	if (key == 'l') glTranslatef(-2.0f, 0.0f, 0.0f);
	if (key == 'k') glTranslatef(0.0f, 2.0f, 0.0f);
	if (key == 'j') glTranslatef(0.0f, -2.0f, 0.0f);

	if (key == 'r') {
		++dimension_order;
		dimension_order %= 3;
		rotate_dimensions();
	}

	float lag = 1.0f;

	if (mods & KMOD_SHIFT) { lag *= 10.0f; }

	if (key == SDLK_F1) proj = PROJECTION_NONE;
	if (key == SDLK_F2) proj = PROJECTION_X;
	if (key == SDLK_F3) proj = PROJECTION_Y;
	if (key == SDLK_F4) proj = PROJECTION_T;
	if (key == SDLK_F5) resetView();

	if (key == SDLK_UP) glScalef(1.5f, 1.5f, 1.5f);
	if (key == SDLK_DOWN) glScalef(0.9f, 0.9f, 0.9f);

	/*
	    // dim dimension_order
	    if (mods & KMOD_SHIFT) {
	        if (key == SDLK_LEFT)  if (dimension_order > 0) dimension_order--;
	        if (key == SDLK_RIGHT) dimension_order++;
	        dimension_order %= 3;
	        rotate_dimensions();
	    }
	*/
	if (proj == PROJECTION_NONE) {
		if (paused) {
			if (key == SDLK_LEFT) timer -= lag;
			if (key == SDLK_RIGHT) timer += lag;
			int sz = func.extent(0);
			if (timer >= sz) timer = sz - 1;
		}
	} else {
		if (key == SDLK_LEFT) glTranslatef(2.0f, 0.0f, 0.0f);
		if (key == SDLK_RIGHT) glTranslatef(-2.0f, 0.0f, 0.0f);
	}
}

int
get_delta_t() {
	return std::floor(1000.0f * delta[0]);
}

Uint32
onTimer(Uint32 interval, void*) {
	if (!paused) timer++;
	if (timer >= func.extent(0)) timer = 0;
	return interval;
}

void
initOpenGL(int argc, char** argv) {
	const int wnd_w = 1280;
	const int wnd_h = 720;
	SDL_CHECK(SDL_Init(SDL_INIT_VIDEO|SDL_INIT_TIMER));
    SDL_CHECK(SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1));
    window = SDL_CreateWindow(
		"ARMA",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		wnd_w,
		wnd_h,
		SDL_WINDOW_OPENGL|SDL_WINDOW_RESIZABLE
    );
    glcontext = SDL_GL_CreateContext(window);
    SDL_CHECK(SDL_GL_SetSwapInterval(1));

	ImGui_SDL_Init(window);

	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	onResize(wnd_w, wnd_h);
	resetView();

	SDL_AddTimer(get_delta_t(), onTimer, nullptr);
}

void
read_valarray(std::istream& in) {
	in >> func;
}

void
read_delta() {
	std::ifstream in("zdelta");
	if (in.is_open()) { in >> delta; }
}

void
parse_cmdline(int argc, char** argv) {
	using namespace std;
	stringstream cmdline;
	for (int i = 1; i < argc; i++) cmdline << argv[i] << ' ';
	string file_name;
	string ar;
	while (!(cmdline >> ar).eof()) {
		if (ar == "-r")
			cmdline >> tail;
		else if (ar == "-t")
			cmdline >> timer;
		else {
			file_name = ar;
		}
		cmdline >> ws;
	}
	if (!file_name.empty()) {
		std::clog << "reading " << ar << std::endl;
		ifstream in(ar.c_str());
		read_valarray(in);
	} else {
		read_valarray(cin);
	}
	read_delta();
	if (timer >= func.extent(0)) { timer = func.extent(0) - 1; }
}

void
main_loop() {
	SDL_Event event;
	bool stopped = false;
	while (!stopped) {
		while (SDL_PollEvent(&event)) {
			switch (event.type) {
				case SDL_QUIT:
					stopped = true;
					break;
				case SDL_WINDOWEVENT:
					switch (event.window.event) {
						case SDL_WINDOWEVENT_SIZE_CHANGED:
							onResize(event.window.data1, event.window.data2);
							break;
						case SDL_WINDOWEVENT_CLOSE:
							event.type = SDL_QUIT;
							SDL_PushEvent(&event);
							break;
					}
					break;
				case SDL_KEYDOWN:
					if (event.key.keysym.mod & KMOD_CTRL) {
						onKeyPressed(event.key.keysym.sym, event.key.keysym.mod);
					}
					break;
				case SDL_MOUSEMOTION:
					if (event.motion.state & SDL_BUTTON_MMASK) {
						onMouseDrag(event.motion.xrel, event.motion.yrel);
					}
					break;
				default:
					break;
			}
			ImGui_SDL_ProcessEvent(&event);
		}
		onDisplay();
	}
}

int
main(int argc, char** argv) {
	#if ARMA_PROFILE
	register_all_counters();
	#endif
	parse_cmdline(argc, argv);
	initOpenGL(argc, argv);
	main_loop();
	ImGui_SDL_Shutdown();
    SDL_GL_DeleteContext(glcontext);
    SDL_DestroyWindow(window);
    SDL_Quit();
	return 0;
}
