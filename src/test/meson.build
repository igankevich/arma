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
	'arma::apmath::Polynomial',
	executable(
		'polynomial-test',
		sources: ['polynomial-test.cc'] + apmath_files,
		include_directories: incdirs,
		dependencies: [
			libblitz,
			gtest,
			libCGAL,
			libgmp,
		]
	)
)

test(
	'arma::apmath::hermite_polynomial',
	executable(
		'hermite-test',
		sources: ['hermite-test.cc'] + apmath_files,
		include_directories: incdirs,
		dependencies: [
			libblitz,
			gtest,
			libCGAL,
			libgmp,
		]
	)
)

test(
	'arma::apmath::owen_t',
	executable(
		'owen-t-test',
		sources: ['owen-t-test.cc'] + apmath_files,
		include_directories: incdirs,
		dependencies: [
			libblitz,
			gtest,
			libCGAL,
			libgmp,
		]
	)
)

test(
	'arma::apmath::Skew_normal',
	executable(
		'skew-normal-test',
		sources: ['skew-normal-test.cc'] + apmath_files,
		include_directories: incdirs,
		dependencies: [
			libblitz,
			libgsl,
			gtest,
			libCGAL,
			libgmp,
		]
	)
)

test(
	'arma::apmath::closed_interval',
	executable(
		'closed-interval-test',
		sources: ['closed-interval-test.cc'] + apmath_files,
		include_directories: incdirs,
		dependencies: [
			libblitz,
			libgsl,
			gtest,
			libCGAL,
			libgmp,
		]
	)
)

test(
	'arma::Output_flags',
	executable(
		'output-flags-test',
		sources: ['output-flags-test.cc'] + output_flags_files,
		include_directories: incdirs,
		dependencies: [gtest]
	)
)

test(
	'arma::apmath::Fourier_transform',
	executable(
		'fourier-test',
		sources: ['fourier-test.cc'] + fourier_src + opencl_src,
		include_directories: incdirs,
		dependencies: [gtest,libblitz,libgsl] + fourier_deps
	)
)

test(
	'arma::apmath::Convolution',
	executable(
		'convolution-test',
		sources: ['convolution-test.cc'] + fourier_src + opencl_src,
		include_directories: incdirs,
		dependencies: [gtest,libblitz,libgsl] + fourier_deps
	)
)
