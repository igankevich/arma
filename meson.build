project(
	'autoreg',
	'cpp',
	version: '0.9',
	meson_version: '>=0.31',
	default_options: ['cpp_std=c++14']
)

if get_option('buildtype') == 'release'
	add_global_arguments('-DNDEBUG', language: 'cpp')
endif

cpp = meson.get_compiler('cpp')
libGL = cpp.find_library('GL')
libblitz = dependency('blitz')
libgsl = dependency('gsl')
libglut = dependency('freeglut')

subdir('src')
