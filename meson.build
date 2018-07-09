project(
	'arma',
	'cpp',
	version: '0.9',
	meson_version: '>=0.31',
	default_options: ['cpp_std=c++14']
)

executable_prefix = 'arma'

cpp = meson.get_compiler('cpp')
libdcmt = cpp.find_library('dcmt')
libblitz = dependency('blitz')
libgsl = dependency('gsl')

if not get_option('buildtype').contains('debug')
	add_global_arguments('-DNDEBUG', language: 'cpp')
else
	# debug Blitz++
	add_global_arguments('-DBZ_DEBUG', language: 'cpp')
endif

arma_deps = [libblitz, libgsl, libdcmt]

if get_option('blas') == 'openblas'
	arma_deps += cpp.find_library('blas')
endif

if get_option('lapack') == 'netlib'
	lapackdep = dependency('lapack-netlib', required: false)
	lapacklib = cpp.find_library('lapack')
	if lapackdep.found()
		arma_deps += lapackdep
	else 
		arma_deps += lapacklib
	endif
	arma_deps += cpp.find_library('lapacke')
	arma_deps += cpp.find_library('gfortran')
endif

if get_option('blas') == 'mkl' or get_option('lapack') == 'mkl'
	if (cpp.get_id() == 'intel')
		if get_option('framework') == 'openmp'
			libmkl = declare_dependency(link_args: ['-mkl=parallel'])
		else
			libmkl = declare_dependency(link_args: '-mkl=sequential')
		endif
	else
		libmkl = declare_dependency(
			link_args: [
				'-lmkl_intel_lp64',
				'-lmkl_intel_thread',
				'-lmkl_core',
				'-liomp5',
				'-lpthread',
				'-lm',
				'-ldl',
			],
		)
	endif
	arma_deps += libmkl
endif

framework = get_option('framework')
if framework == 'openmp'
	add_global_arguments('-DARMA_OPENMP=1', language: 'cpp')
	if cpp.get_id() == 'gcc'
		add_global_arguments('-fopenmp', language: 'cpp')
		add_global_link_arguments('-fopenmp', language: 'cpp')
	elif cpp.get_id() == 'intel'
		add_global_arguments('-qopenmp', language: 'cpp')
		add_global_link_arguments('-qopenmp', language: 'cpp')
	endif
	# make Blitz++ reference counters thread-safe
	add_global_arguments('-DBZ_THREADSAFE', language: 'cpp')
elif framework == 'opencl'
	add_global_arguments('-DARMA_OPENCL=1', language: 'cpp')
	arma_deps += [
		dependency('OpenCL'),
		dependency('threads'),
		dependency('clFFT'),
	]
elif framework == 'mpi'
	add_global_arguments('-DARMA_MPI=1', language: 'cpp')
elif framework == 'bscheduler'
	add_global_arguments('-DARMA_BSCHEDULER=1', language: 'cpp')
	# make Blitz++ reference counters thread-safe
	add_global_arguments('-DBZ_THREADSAFE', language: 'cpp')
	arma_deps += [
		dependency('bscheduler-app'),
		dependency('threads'),
	]
else
	add_global_arguments('-DARMA_NONE=1', language: 'cpp')
endif

add_global_arguments(
	'-DARMA_REAL_TYPE=' + get_option('real_type'),
	language: 'cpp'
)

if get_option('profile')
	add_global_arguments('-DARMA_PROFILE=1', language: 'cpp')
endif

run_target('graphs', command: [
	'gnuplot',
	'--persist',
	'-c',
	'scripts/target-graphs',
	meson.build_root()
])
run_target('slices', command: 'scripts/target-slices')
run_target('velocity', command: 'scripts/target-velocity.R')
run_target('archive', command: [
	'scripts/target-archive',
	meson.project_name(),
	meson.project_version()
])
foreach testcase : ['standing', 'propagating']
	run_target('nit-' + testcase, command: [
		'scripts/target-nit',
		'nit-' + testcase + '-none',
		'nit-' + testcase + '-gramcharlier',
		'nit-' + testcase + '-skewnormal'
	])
endforeach
run_target('nit-plot', command: 'scripts/target-nit.R')
run_target('propagating-wave-slice', command: 'scripts/target-propagating-wave-slice.R')

datadir = get_option('prefix') + '/' + get_option('datadir')
opencl_srcdir = get_option('opencl_srcdir')
if opencl_srcdir.strip() == ''
	opencl_srcdir = datadir + '/' + meson.project_name() + '/' + 'kernels'
endif
config = configuration_data()
config.set('mt_config_file', 'mt.dat')
config.set('datadir', datadir)
config.set('opencl_srcdir', opencl_srcdir)
config.set('ARMA_CLFFT', get_option('framework') == 'opencl')
config.set('ARMA_HART_SOLVER', get_option('with_high_amplitude_realtime_solver'))
config.set('ARMA_SIMULATE_FAILURES', get_option('simulate_failures'))
if cpp.has_header('lapacke.h')
	config.set('ARMA_LAPACKE_H', true)
elif cpp.has_header('lapacke/lapacke.h')
	config.set('ARMA_LAPACKE_LAPACKE_H', true)
endif
if cpp.has_header('cblas.h')
	config.set('ARMA_CBLAS_H', true)
elif cpp.has_header('openblas/cblas.h')
	config.set('ARMA_OPENBLAS_CBLAS_H', true)
endif

src = include_directories('src')
subdir('src')

# doxygen {{{
doxygen = find_program('doxygen', required: false)
if doxygen.found()
	run_target(
		'doc',
		command: [doxygen, join_paths(meson.source_root(), 'Doxyfile')]
	)
endif
# }}}
