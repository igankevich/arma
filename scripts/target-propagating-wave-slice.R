#!/usr/bin/Rscript

middle_element <- function (x) {
	x[round(length(x) + 0.5)*2/3]
}

# go to build directory
build_dir <- Sys.getenv('MESON_BUILD_ROOT')
if (!is.null(build_dir) & build_dir != '') {
	setwd(build_dir)
}

read_zeta_slice <- function (filename, offset_t=0) {
	# slice time and Y ranges through the center
	zeta <- read.csv(filename)
	slice_t <- middle_element(unique(zeta$t)) + offset_t
	slice_y <- middle_element(unique(zeta$y))

	# get wave profile
	slice_y <- middle_element(unique(zeta$y))
#	print(paste('Middle elements of zeta (TY) = ', slice_t, slice_y))
	zeta[zeta$t == slice_t & zeta$y == slice_y,]
}

cairo_pdf(filename="propagating-wave-slice.pdf", height=10, width=7)
noffsets <- 5
stride <- 2
amp <- 1
r <- 3*sqrt(amp)
all_offsets <- seq(1, stride*noffsets, by=stride) - 1
par(mfrow=c(noffsets, 1))
for (offset in all_offsets) {
	message(paste("plot offset", offset))
	zeta <- read_zeta_slice('zeta.csv', offset)
	plot.new()
	plot.window(xlim=range(zeta$x), ylim=range(c(-r, r)), asp=1)
	axis(1)
	axis(2)
	lines(zeta$x, zeta$z, lty='solid')
	title(xlab="x", ylab="z")
	box()
}

