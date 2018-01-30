gtest = dependency('gtest', main: true)

test(
	'linalg::*',
	executable(
		'linalg-test',
		sources: ['linalg-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma],
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
	'arma::apmath::Polynomial',
	executable(
		'polynomial-test',
		sources: ['polynomial-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma]
	)
)

test(
	'arma::apmath::hermite_polynomial',
	executable(
		'hermite-test',
		sources: ['hermite-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma]
	)
)

test(
	'arma::apmath::owen_t',
	executable(
		'owen-t-test',
		sources: ['owen-t-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma]
	)
)

test(
	'arma::apmath::Skew_normal',
	executable(
		'skew-normal-test',
		sources: ['skew-normal-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma]
	)
)

test(
	'arma::apmath::closed_interval',
	executable(
		'closed-interval-test',
		sources: ['closed-interval-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma]
	)
)

test(
	'arma::Output_flags',
	executable(
		'output-flags-test',
		sources: ['output-flags-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma]
	)
)

test(
	'arma::apmath::Fourier_transform',
	executable(
		'fourier-test',
		sources: ['fourier-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma]
	)
)

test(
	'arma::apmath::Convolution',
	executable(
		'convolution-test',
		sources: ['convolution-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma]
	)
)

test(
	'arma::Yule_walker_solver',
	executable(
		'yule-walker-test',
		sources: ['yule-walker-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma],
	)
)

test(
	'arma::auto_covariance',
	executable(
		'auto-covariance-test',
		sources: ['auto-covariance-test.cc'],
		include_directories: incdirs,
		dependencies: [gtest, libarma],
	)
)
