executable(
	'arma',
	sources: 'arma.cc',
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
	'dcmt',
	sources: 'dcmt.cc',
	dependencies: [libdcmt]
)

executable(
	'visual',
	sources: 'visual.cc',
	dependencies: [
		libblitz,
		libglut,
		libGL
	]
)

executable(
	'slice',
	sources: 'slice.cc',
	dependencies: [libblitz]
)

test(
	'linalg::*',
	executable(
		'linalg-test',
		sources: 'linalg-test.cc',
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
