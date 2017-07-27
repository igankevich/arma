arma_dcmt_name = executable_prefix + '-dcmt'
config.set('arma_dcmt_name', arma_dcmt_name)

configure_file(
	input: 'config.hh.in',
	output: 'config.hh',
	configuration: config
)

linalg_files = files([
	'linalg.cc',
	'params.cc',
	'apmath/closed_interval.cc',
])
apmath_files = files([
	'apmath/polynomial.cc',
	'apmath/hermite.cc',
	'apmath/owen_t.cc',
	'apmath/closed_interval.cc',
])
nonlinear_files = files([
	'nonlinear/series.cc',
	'nonlinear/nit_transform.cc',
])

common_src = [
	'arma.cc',
	'params.cc',
	'bits/acf_wrapper.cc',
	'generator/plain_wave_model.cc',
	'generator/ar_model.cc',
	'generator/ma_model.cc',
	'generator/arma_model.cc',
	'generator/lh_model.cc',
	'generator/basic_arma_model.cc',
	'ma_algorithm.cc',
	'interpolate.cc',
	'verification_scheme.cc',
	'stats/qq_graph.cc',
	'stats/summary.cc',
	'stats/waves.cc',
	'stats/distribution.cc',
	'stats/gaussian.cc',
	'stats/gram_charlier.cc',
	'stats/skew_normal.cc',
] + linalg_files + apmath_files + nonlinear_files

if get_option('profile')
	common_src += 'profile.cc'
endif

if get_option('framework') == 'opencl'
	common_src += [
		'opencl/opencl.cc',
		'opencl/gl_object_guard.cc',
		'opencl/device_type.cc',
	]
endif

arma_src = common_src + [
	'velocity/basic_solver.cc',
	'velocity/plain_wave_solver.cc',
	'velocity/linear_solver.cc',
	'velocity/high_amplitude_solver.cc',
]

arma_cpp_args = []
arma_deps = [
	libblitz,
	libgsl,
	libblas,
	liblapack
]

if get_option('with_small_amplitude_solver')
	arma_src += [
		'velocity/small_amplitude_solver.cc',
		'velocity/small_amplitude_solver/wave.cc',
		'velocity/small_amplitude_solver/linear_interpolation.cc',
	]
	arma_cpp_args += '-DWITH_SMALL_AMPLITUDE_SOLVER' 
	arma_deps += [libCGAL, libgmp]
endif

if get_option('framework') == 'opencl'
	arma_deps += [libOpenCL, libpthread]
endif

executable(
	executable_prefix,
	sources: arma_src + ['arma_main.cc'],
	include_directories: incdirs,
	dependencies: arma_deps + parallel_deps,
	cpp_args: arma_cpp_args,
	link_args: '-llapacke',
	install: true
)

if get_option('with_high_amplitude_realtime_solver')
	arma_realtime_src = common_src + [
		'velocity/basic_solver.cc',
		'velocity/high_amplitude_realtime_solver.cc',
	]
	arma_realtime_cpp_args = arma_cpp_args + ['-DARMA_OPENGL=1']
	arma_realtime_deps = arma_deps + [libglut, libGLU, libGL, libclFFT] + parallel_deps
	executable(
		executable_prefix + '-realtime',
		sources: arma_realtime_src + [
			'arma_realtime.cc',
			'arma_realtime_driver.cc'
		],
		include_directories: incdirs,
		dependencies: arma_realtime_deps,
		cpp_args: arma_realtime_cpp_args,
		link_args: '-llapacke',
		install: true
	)
endif

executable(
	arma_dcmt_name,
	sources: 'dcmt.cc',
	dependencies: [libdcmt],
	install: true
)

executable(
	executable_prefix + '-visual',
	sources: 'visual.cc',
	dependencies: [
		libblitz,
		libglut,
		libGL
	],
	install: true
)

executable(
	'layout-run',
	sources: 'layout-run.cc'
)

subdir('test')
