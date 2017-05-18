# Developer build with OpenCL

	meson . build
	cd build
	mesonconf -Dframework=opencl -Dopencl_srcdir=../src/kernels
	ninja

