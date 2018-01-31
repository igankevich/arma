arma_dcmt_name = executable_prefix + '-dcmt'
config.set('arma_dcmt_name', arma_dcmt_name)

configure_file(
	input: 'config.hh.in',
	output: 'config.hh',
	configuration: config
)

imgui_src = []
arma_lib_src = []

arma_cpp_args = []

subdir('bits')
subdir('imgui')
subdir('apmath')
subdir('nonlinear')
subdir('stats')
subdir('generator')
subdir('velocity')
subdir('opencl')

arma_lib_src += files([
	'arma.cc',
	'arma_driver.cc',
	'chop.cc',
	'factor_waves.cc',
	'interpolate.cc',
	'linalg.cc',
	'ma_algorithm.cc',
	'output_flags.cc',
	'parallel_mt.cc',
	'params.cc',
	'params.cc',
	'util.cc',
	'wave.cc',
	'yule_walker.cc',
])

if get_option('profile')
	arma_lib_src += 'profile.cc'
endif

arma_lib = shared_library(
	'arma',
	sources: arma_lib_src,
	include_directories: incdirs,
	dependencies: arma_deps,
	cpp_args: arma_cpp_args,
	link_args: '-llapacke',
	version: meson.project_version(),
	install: true
)

libarma = declare_dependency(
	compile_args: arma_cpp_args,
	link_with: arma_lib,
	link_args: '-llapacke',
	include_directories: incdirs,
	dependencies: arma_deps
)

executable(
	executable_prefix,
	sources: 'arma_main.cc',
	include_directories: incdirs,
	dependencies: [libarma],
	install: true
)

if get_option('with_high_amplitude_realtime_solver')
	arma_realtime_src = arma_lib_src + [
		'velocity/basic_solver.cc',
		'velocity/high_amplitude_realtime_solver.cc',
	]
	arma_realtime_cpp_args = arma_cpp_args + ['-DARMA_OPENGL=1']
	arma_realtime_deps = arma_deps + [libglut, libGL, libclFFT]
	executable(
		executable_prefix + '-realtime',
		sources: arma_realtime_src + [
			'arma_realtime.cc',
			'arma_realtime_driver.cc'
		],
		include_directories: incdirs,
		dependencies: arma_realtime_deps,
		cpp_args: arma_realtime_cpp_args,
		link_args: '-llapacke',
		install: true
	)
endif

executable(
	arma_dcmt_name,
	sources: 'dcmt.cc',
	dependencies: [libdcmt],
	install: true
)

executable(
	executable_prefix + '-visual',
	sources: ['visual.cc'] + imgui_src,
	dependencies: [libGL, libSDL2, libarma, dependency('threads')],
	install: true
)

subdir('test')
