#!/usr/bin/Rscript

middle_element <- function (x) {
	x[round(length(x) + 0.5)/2]
}

# go to build directory
build_dir <- Sys.getenv('MESON_BUILD_ROOT')
if (!is.null(build_dir) & build_dir != '') {
	setwd(build_dir)
}

# slice time and Y ranges through the center
zeta <- read.csv('zeta.csv')
slice_t <- middle_element(unique(zeta$t))
slice_y <- middle_element(unique(zeta$y))

# get wave profile
slice_y <- middle_element(unique(zeta$y))
print(paste('Middle elements of zeta (TY) = ', slice_t, slice_y))
zeta_slice <- zeta[zeta$t == slice_t & zeta$y == slice_y,]

x <- unique(zeta_slice$x)
z <- unique(zeta_slice$z)

# plot the graph
rx <- range(x)
rz <- range(z)
aspect_ratio <- (rx[[2]] - rx[[1]]) / (rz[[2]] - rz[[1]])
print(aspect_ratio)
aspect_ratio <- 1
cairo_pdf(filename="nonlinear.pdf", width=2.5*aspect_ratio, height=5)
plot(zeta_slice$x, zeta_slice$z, lwd=4, asp=1)
