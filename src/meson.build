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
	'apmath/closed_interval.cc',
	'apmath/delaunay_interpolation.cc',
	'apmath/hermite.cc',
	'apmath/owen_t.cc',
	'apmath/polynomial.cc',
])
nonlinear_files = files([
	'nonlinear/series.cc',
	'nonlinear/nit_transform.cc',
])

output_flags_files = files([
	'output_flags.cc'
])

common_src = files([
	'arma.cc',
	'params.cc',
	'parallel_mt.cc',
	'bits/acf_wrapper.cc',
	'generator/plain_wave_model.cc',
	'generator/ar_model.cc',
	'generator/ma_model.cc',
	'generator/arma_model.cc',
	'generator/lh_model.cc',
	'generator/basic_arma_model.cc',
	'generator/voodoo.cc',
	'ma_algorithm.cc',
	'interpolate.cc',
	'stats/qq_graph.cc',
	'stats/summary.cc',
	'stats/waves.cc',
	'stats/distribution.cc',
	'stats/gaussian.cc',
	'stats/gram_charlier.cc',
	'stats/skew_normal.cc',
	'arma_driver.cc',
	'util.cc'
]) + linalg_files + apmath_files + nonlinear_files + output_flags_files

if get_option('profile')
	common_src += 'profile.cc'
endif

opencl_src = []
if get_option('framework') == 'opencl'
	opencl_src = files([
		'opencl/opencl.cc',
		'opencl/gl_object_guard.cc',
		'opencl/device_type.cc',
		'params.cc'
	])
	common_src += opencl_src
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
	liblapack,
	libdcmt,
	libCGAL,
	libgmp,
]

if get_option('with_small_amplitude_solver')
	arma_src += [
		'velocity/small_amplitude_solver.cc',
		'velocity/small_amplitude_solver/wave.cc',
	]
	arma_cpp_args += '-DWITH_SMALL_AMPLITUDE_SOLVER' 
endif

fourier_src = []
fourier_deps = []
if get_option('framework') == 'opencl'
	fourier_src = files(['apmath/fourier_opencl.cc'])
	fourier_deps = [libOpenCL, libpthread, libclFFT]
	arma_deps += fourier_deps
	arma_src += fourier_src
endif

executable(
	executable_prefix,
	sources: arma_src + ['arma_main.cc'],
	include_directories: incdirs,
	dependencies: arma_deps,
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
	arma_realtime_deps = arma_deps + [libglut, libGLU, libGL, libclFFT]
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
	] + fourier_deps,
	install: true
)

subdir('test')
