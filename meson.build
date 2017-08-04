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
liblapack = cpp.find_library('lapack')
libblas = cpp.find_library('blas')
libGL = cpp.find_library('GL')
libglut = dependency('freeglut')
libGLU = cpp.find_library('GLU')
gtest = dependency('gtest', main: true)
libCGAL = cpp.find_library(
	'CGAL',
	required: get_option('with_small_amplitude_solver')
)
libgmp = cpp.find_library(
	'gmp',
	required: get_option('with_small_amplitude_solver')
)
libOpenCL = dependency(
	'OpenCL',
	required: get_option('framework') == 'opencl'
)
libpthread = dependency(
	'threads',
	required: get_option('framework') == 'opencl'
)
libclFFT = dependency(
	'clFFT',
	required: get_option('with_high_amplitude_realtime_solver')
)

if get_option('buildtype') == 'release'
	add_global_arguments('-DNDEBUG', language: 'cpp')
else
	# debug Blitz++
	add_global_arguments('-DBZ_DEBUG', language: 'cpp')
endif

framework = get_option('framework')
if framework == 'openmp'
	add_global_arguments('-DARMA_OPENMP=1', language: 'cpp')
	add_global_arguments('-fopenmp', language: 'cpp')
	# make Blitz++ reference counters thread-safe
	add_global_arguments('-DBZ_THREADSAFE', language: 'cpp')
	add_global_link_arguments('-fopenmp', language: 'cpp')
elif framework == 'opencl'
	add_global_arguments('-DARMA_OPENCL=1', language: 'cpp')
elif framework == 'mpi'
	add_global_arguments('-DARMA_MPI=1', language: 'cpp')
else
	add_global_arguments('-DARMA_NONE=1', language: 'cpp')
endif

if get_option('noseed')
	add_global_arguments('-DARMA_NO_PRNG_SEED', language: 'cpp')
endif

add_global_arguments(
	'-DARMA_REAL_TYPE=' + get_option('real_type'),
	language: 'cpp'
)

if get_option('profile')
	add_global_arguments('-DARMA_PROFILE=1', language: 'cpp')
endif

run_target('doc', command: 'scripts/generate-docs')
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

datadir = get_option('prefix') + '/' + get_option('datadir')
opencl_srcdir = get_option('opencl_srcdir')
if opencl_srcdir.strip() == ''
	opencl_srcdir = datadir + '/' + meson.project_name() + '/' + 'kernels'
endif
config = configuration_data()
config.set('mt_config_file', 'mt.dat')
config.set('datadir', datadir)
config.set('opencl_srcdir', opencl_srcdir)

incdirs = include_directories('src')
subdir('src')
