# Build

	mkdir build
	meson . build
	ninja -C build

# Run

	cd build
	./src/autoreg -c standind_wave.autoreg # generate wavy surface
	./visual zeta                          # visualise wavy surface
