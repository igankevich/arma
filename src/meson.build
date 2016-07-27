executable(
	'autoreg',
	sources: 'autoreg.cc',
	dependencies: [
		libblitz,
		libgsl
	]
)

libdcmt_include = include_directories('dcmt0.6.2/include')

libdcmt = declare_dependency(
	include_directories: libdcmt_include,
	link_args: ['-L'+meson.current_source_dir()+'/dcmt0.6.2/lib', '-ldcmt']
)


executable(
	'dcmt',
	sources: 'dcmt.cc',
	dependencies: [
		libdcmt
	]
)

executable(
	'visual',
	sources: 'visual.cc',
	dependencies: [
		libblitz,
		libglut,
		libGL
	]
)
