# Build

	mkdir build
	meson . build
	ninja -C build

# Run

	cd build
	./src/arma -c standind_wave.arma # generate wavy surface
	./visual zeta                    # visualise wavy surface

# Developer build with OpenCL

	meson . build
	cd build
	mesonconf -Dframework=opencl -Dopencl_srcdir=../src/kernels
	ninja


# Build with MKL

	meson -Dblas=mkl -Dlapack=mkl . build

