arma_lib_src += files([
	'closed_interval.cc',
	'hermite.cc',
	'owen_t.cc',
	'polynomial.cc',
])

if get_option('framework') == 'opencl'
	arma_lib_src += files(['fourier_opencl.cc'])
endif
