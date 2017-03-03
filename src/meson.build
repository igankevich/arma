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
	dependencies: [
		libdcmt
	]
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
	dependencies: [ libblitz ]
)

test(
	'cholesky decomposition',
	executable(
		'cholesky-test',
		sources: 'cholesky-test.cc',
		dependencies: [
			libblitz,
			libgsl,
			libblas,
			liblapack
		],
		link_args: '-llapacke'
	)
)

test(
	'inverse matrix',
	executable(
		'inverse-test',
		sources: 'inverse-test.cc',
		dependencies: [
			libblitz,
			libgsl,
			libblas,
			liblapack
		],
		link_args: '-llapacke'
	)
)

test(
	'matrix vector multiply',
	executable(
		'multiply-test',
		sources: 'multiply-test.cc',
		dependencies: [
			libblitz,
			libgsl,
			libblas,
			liblapack,
			libgsl
		],
		link_args: '-llapacke'
	)
)

test(
	'dynamic creation of mersenne twisters',
	executable(
		'dcmt-test',
		sources: 'dcmt-test.cc',
		dependencies: [ libdcmt ]
	)
)

test(
	'arma::Domain',
	executable(
		'domain-test',
		sources: 'domain-test.cc',
		dependencies: [libblitz]
	)
)
