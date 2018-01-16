arma_lib_src += files([
	'basic_solver.cc',
	'high_amplitude_solver.cc',
	'linear_solver.cc',
	'plain_wave_solver.cc',
])

if get_option('with_small_amplitude_solver')
	arma_lib_src += [
		'delaunay_interpolation.cc',
		'small_amplitude_solver.cc',
	]
	arma_cpp_args += '-DWITH_SMALL_AMPLITUDE_SOLVER' 
	arma_deps += [
		cpp.find_library('CGAL'),
		cpp.find_library('gmp')
	]
endif

