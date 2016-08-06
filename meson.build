project(
	'autoreg',
	'cpp',
	version: '0.9',
	meson_version: '>=0.31',
	default_options: ['cpp_std=c++14']
)

meson.add_postconf_script('scripts/copy-input-files')

if get_option('buildtype') == 'release'
	add_global_arguments('-DNDEBUG', language: 'cpp')
endif

run_target('doc', 'scripts/generate-docs')
run_target('graphs', 'gnuplot', '--persist', '-c', 'scripts/target-graphs', meson.build_root())

cpp = meson.get_compiler('cpp')
libdcmt = cpp.find_library('dcmt')
libblitz = dependency('blitz')
libgsl = dependency('gsl')
liblapack = cpp.find_library('lapack')
libblas = cpp.find_library('blas')
libGL = cpp.find_library('GL')
libglut = dependency('freeglut')

subdir('src')
