project(
	'arma',
	'cpp',
	version: '0.9',
	meson_version: '>=0.31',
	default_options: ['cpp_std=c++14']
)

meson.add_postconf_script('scripts/copy-input-files')

cpp = meson.get_compiler('cpp')
libdcmt = cpp.find_library('dcmt')
libblitz = dependency('blitz')
libgsl = dependency('gsl')
liblapack = cpp.find_library('lapack')
libblas = cpp.find_library('blas')
libGL = cpp.find_library('GL')
libglut = dependency('freeglut')
gtest = dependency('gtest', main: true)

if get_option('buildtype') == 'release'
	add_global_arguments('-DNDEBUG', language: 'cpp')
endif

parallel_deps = []
framework = get_option('framework')
if framework == 'openmp'
	add_global_arguments('-DARMA_OPENMP=1', language: 'cpp')
	add_global_arguments('-fopenmp', language: 'cpp')
	add_global_link_arguments('-fopenmp', language: 'cpp')
	parallel_deps += libdcmt
elif framework == 'opencl'
	add_global_arguments('-DARMA_OPENCL=1', language: 'cpp')
	add_global_link_arguments('-lOpenCL', language: 'cpp')
elif framework == 'mpi'
	add_global_arguments('-DARMA_MPI=1', language: 'cpp')
else
	add_global_arguments('-DARMA_NONE=1', language: 'cpp')
endif

if get_option('noseed')
	add_global_arguments('-DARMA_NO_PRNG_SEED', language: 'cpp')
endif

run_target('doc', command: 'scripts/generate-docs')
run_target('graphs', command: ['gnuplot', '--persist', '-c', 'scripts/target-graphs', meson.build_root()])
run_target('slices', command: 'scripts/target-slices')
run_target('velocity', command: 'scripts/target-velocity.R')

incdirs = include_directories('src')
subdir('src')
