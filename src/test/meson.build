gtest_nomain = dependency('gtest', main: false)
gtest_main = dependency('gtest', main: true)

arma_test_lib = shared_library(
	'arma_test_main',
	sources: ['arma_test_main.cc', 'arma_environment.cc'],
	dependencies: [gtest_nomain,libarma],
	include_directories: src
)

arma_test_main = declare_dependency(
	link_with: [arma_test_lib],
	dependencies: [libarma,gtest_nomain],
	include_directories: src
)


all_tests = [
	['arma::prng::parallel_mt', 'dcmt-test', [gtest_main,libdcmt]],
	['arma::Domain', 'domain-test', [gtest_main,libblitz]],
	['arma::Grid', 'grid-test', [gtest_main,libblitz]],
	['arma::apmath::factorial', 'factorial-test', [gtest_main,libblitz]],
	['linalg::*', 'linalg-test', [arma_test_main]],
	['arma::apmath::Polynomial', 'polynomial-test', [arma_test_main]],
	['arma::apmath::hermite_polynomial', 'hermite-test', [arma_test_main]],
	['arma::apmath::owen_t', 'owen-t-test', [arma_test_main]],
	['arma::apmath::Skew_normal', 'skew-normal-test', [arma_test_main]],
	['arma::apmath::closed_interval', 'closed-interval-test', [arma_test_main]],
	['arma::Output_flags', 'output-flags-test', [arma_test_main]],
	['arma::apmath::Fourier_transform', 'fourier-test', [arma_test_main]],
	['arma::apmath::Convolution', 'convolution-test', [arma_test_main]],
	['arma::Yule_walker_solver', 'yule-walker-test', [arma_test_main]],
	['arma::auto_covariance', 'auto-covariance-test', [arma_test_main]],
	['arma::generator::acf_generator', 'acf-generator-test', [arma_test_main]],
	['arma::stats::waves', 'factor-waves-test', [arma_test_main]],
	['arma::stats::MA_coefficient_solver', 'ma-coefficient-solver-test', [arma_test_main]],
	['arma::derivatives', 'derivatives-test', [arma_test_main]],
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
			dependencies: deps
		)
	)
endforeach

