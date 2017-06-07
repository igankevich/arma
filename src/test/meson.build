test(
	'linalg::*',
	executable(
		'linalg-test',
		sources: ['linalg-test.cc'] + linalg_files,
		include_directories: incdirs,
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
	'arma::prng::parallel_mt',
	executable(
		'dcmt-test',
		sources: 'dcmt-test.cc',
		include_directories: incdirs,
		dependencies: [libdcmt, gtest]
	)
)

test(
	'arma::Domain',
	executable(
		'domain-test',
		sources: 'domain-test.cc',
		include_directories: incdirs,
		dependencies: [libblitz, gtest]
	)
)

test(
	'arma::Grid',
	executable(
		'grid-test',
		sources: 'grid-test.cc',
		include_directories: incdirs,
		dependencies: [libblitz, gtest]
	)
)

test(
	'arma::apmath::factorial',
	executable(
		'factorial-test',
		sources: 'factorial-test.cc',
		include_directories: incdirs,
		dependencies: [libblitz, gtest]
	)
)

test(
	'arma::apmath::polynomial',
	executable(
		'polynomial-test',
		sources: ['polynomial-test.cc'] + apmath_files,
		include_directories: incdirs,
		dependencies: [libblitz, gtest]
	)
)

test(
	'arma::apmath::hermite_polynomial',
	executable(
		'hermite-test',
		sources: ['hermite-test.cc'] + apmath_files,
		include_directories: incdirs,
		dependencies: [libblitz, gtest]
	)
)

test(
	'arma::apmath::owen_t',
	executable(
		'owen-t-test',
		sources: ['owen-t-test.cc'] + apmath_files,
		include_directories: incdirs,
		dependencies: [libblitz, gtest]
	)
)

test(
	'arma::apmath::Skew_normal',
	executable(
		'skew-normal-test',
		sources: ['skew-normal-test.cc'] + apmath_files,
		include_directories: incdirs,
		dependencies: [libblitz, libgsl, gtest]
	)
)
