executable(
	'autoreg',
	sources: 'autoreg.cc',
	dependencies: [
		libblitz,
		libgsl,
		libblas,
		liblapack
	],
	link_args: '-llapacke'
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

test(
	'cholesky decomposition',
	executable(
		'cholesky-test',
		sources: 'cholesky-test.cc',
		dependencies: [
			libblitz,
			libblas,
			liblapack
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
