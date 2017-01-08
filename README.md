# Build

	mkdir build
	meson . build
	ninja -C build

# Run

	cd build
	./src/arma -c standind_wave.arma # generate wavy surface
	./visual zeta                    # visualise wavy surface
