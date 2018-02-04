gtest = dependency('gtest', main: true)

all_tests = [
	['arma::prng::parallel_mt', 'dcmt-test', [libdcmt]],
	['arma::Domain', 'domain-test', [libblitz]],
	['arma::Grid', 'grid-test', [libblitz]],
	['arma::apmath::factorial', 'factorial-test', [libblitz]],
	['linalg::*', 'linalg-test', [libarma]],
	['arma::apmath::Polynomial', 'polynomial-test', [libarma]],
	['arma::apmath::hermite_polynomial', 'hermite-test', [libarma]],
	['arma::apmath::owen_t', 'owen-t-test', [libarma]],
	['arma::apmath::Skew_normal', 'skew-normal-test', [libarma]],
	['arma::apmath::closed_interval', 'closed-interval-test', [libarma]],
	['arma::Output_flags', 'output-flags-test', [libarma]],
	['arma::apmath::Fourier_transform', 'fourier-test', [libarma]],
	['arma::apmath::Convolution', 'convolution-test', [libarma]],
	['arma::Yule_walker_solver', 'yule-walker-test', [libarma]],
	['arma::auto_covariance', 'auto-covariance-test', [libarma]],
	['arma::generator::acf_generator', 'acf-generator-test', [libarma]],
	['arma::stats::waves', 'factor-waves-test', [libarma]],
]

foreach t : all_tests
	name = t[0]
	filename = t[1]
	deps = t[2]
	test(
		name,
		executable(
			filename,
			sources: [filename+'.cc'],
			include_directories: src,
			dependencies: [gtest] + deps
		)
	)
endforeach

