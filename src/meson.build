common_src = [
	'arma.cc',
	'params.cc',
	'linalg.cc',
	'generator/plain_wave.cc',
	'generator/ar_model.cc',
	'generator/ma_model.cc',
	'generator/arma_model.cc',
	'ma_algorithm.cc',
]

arma_src = common_src + [
	'velocity/basic_solver.cc',
	'velocity/plain_wave_solver.cc',
	'velocity/linear_solver.cc',
	'velocity/high_amplitude_solver.cc',
]

arma_realtime_src = common_src + [
	'velocity/basic_solver.cc',
	'velocity/high_amplitude_realtime_solver.cc',
]

executable(
	executable_prefix,
	sources: arma_src + ['arma_main.cc'],
	include_directories: incdirs,
	dependencies: [
		libblitz,
		libgsl,
		libblas,
		liblapack
	] + parallel_deps,
	link_args: '-llapacke',
	install: true
)

executable(
	executable_prefix + '-realtime',
	sources: arma_realtime_src + ['arma_realtime.cc'],
	include_directories: incdirs,
	dependencies: [
		libblitz,
		libgsl,
		libblas,
		liblapack,
		libglut,
		libGL
	] + parallel_deps,
	link_args: '-llapacke',
	install: true
)


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
