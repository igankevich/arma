#!/usr/bin/Rscript

middle_element <- function (x) {
	x[round(length(x) + 0.5)*2/3]
}

# go to build directory
build_dir <- Sys.getenv('MESON_BUILD_ROOT')
if (!is.null(build_dir) & build_dir != '') {
	setwd(build_dir)
}

read_zeta_slice <- function (filename) {
	# slice time and Y ranges through the center
	zeta <- read.csv(filename)
	slice_t <- middle_element(unique(zeta$t))
	slice_y <- middle_element(unique(zeta$y))

	# get wave profile
	slice_y <- middle_element(unique(zeta$y))
	print(paste('Middle elements of zeta (TY) = ', slice_t, slice_y))
	zeta[zeta$t == slice_t & zeta$y == slice_y,]
}

zeta_linear <- read_zeta_slice('zeta-linear.csv')
zeta_nonlinear <- read_zeta_slice('zeta-nonlinear.csv')

x <- unique(zeta_linear$x)
z <- unique(zeta_linear$z)

# plot the graph
rx <- range(x)
rz <- range(z)
aspect_ratio <- (rx[[2]] - rx[[1]]) / (rz[[2]] - rz[[1]])
print(aspect_ratio)
aspect_ratio <- 1
cairo_pdf(filename="nonlinear.pdf", height=2.5*aspect_ratio, width=5)
plot.new()
plot.window(xlim=rx, ylim=rz, asp=1)
axis(1)
axis(2)
lines(zeta_linear$x, zeta_linear$z, lty='dashed')
lines(zeta_nonlinear$x, zeta_nonlinear$z, lty='solid')
title(xlab="x", ylab="z")
box()
