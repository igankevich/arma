common_src = [
	'arma.cc',
	'params.cc',
	'linalg.cc',
	'generator/plain_wave.cc',
	'generator/ar_model.cc',
	'generator/ma_model.cc',
	'generator/arma_model.cc',
	'generator/lh_model.cc',
	'ma_algorithm.cc',
	'interpolate.cc',
]

if get_option('framework') == 'opencl'
	common_src += [
		'opencl/opencl.cc',
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
	arma_depn += [libCGAL, libgmp]
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
	arma_realtime_deps = arma_deps + [libglut, libGL, libclFFT] + parallel_deps
	executable(
		executable_prefix + '-realtime',
		sources: arma_realtime_src + ['arma_realtime.cc'],
		include_directories: incdirs,
		dependencies: arma_realtime_deps,
		link_args: '-llapacke',
		install: true
	)
endif


executable(
	executable_prefix + '-dcmt',
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

test(
	'linalg::*',
	executable(
		'linalg-test',
		sources: ['linalg.cc', 'linalg-test.cc'],
		dependencies: [
			libblitz,
			libgsl,
			libblas,
			liblapack,
			gtest
		],
		link_args: '-llapacke'
	)
)

test(
	'arma::parallel_mt',
	executable(
		'dcmt-test',
		sources: 'dcmt-test.cc',
		dependencies: [libdcmt, gtest]
	)
)

test(
	'arma::Domain',
	executable(
		'domain-test',
		sources: 'domain-test.cc',
		dependencies: [libblitz, gtest]
	)
)
