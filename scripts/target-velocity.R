#!/usr/bin/Rscript

middle_element <- function (x) {
	x[round(length(x) + 0.5)/2]
}

build_dir <- Sys.getenv('MESON_BUILD_ROOT')
if (!is.null(build_dir) & build_dir != '') {
	setwd(build_dir)
}
phi <- read.csv('phi.csv')
left_top_x <- 0.1
right_top_x <- max(phi$x)
# slice time and Y ranges through the center
slice_t <- middle_element(unique(phi$t))
slice_y <- middle_element(unique(phi$y))
print(paste('Middle elements (TY) = ', slice_t, slice_y))
phi_slice <- phi[phi$t == slice_t & phi$y == slice_y & phi$x >= left_top_x & phi$z >= -8,]
x <- unique(phi_slice$x)
z <- unique(phi_slice$z)
left_top_z <- max(phi_slice$z)
right_top_z <- left_top_z
print(paste('Velocity field size (XZ) = ', length(x), length(z)))

# convert data frame to matrix
seq_x <- seq_along(x)
seq_z <- seq_along(z)
indices <- data.matrix(expand.grid(seq_x, seq_z))
u <- with(phi_slice, {
	out <- matrix(nrow=length(seq_x), ncol=length(seq_z))
	out[indices] <- phi
	out
})
#summary(u)

# get wave profile
zeta <- read.csv('zeta.csv')
zeta_slice <- zeta[zeta$t == slice_t & zeta$y == slice_y & zeta$x >= left_top_x,]
#print(zeta_slice)
#print(x)


# plot the image
require(grDevices)

cairo_pdf(filename="u.pdf")
#par(pty="s")
contour(
	x, z, u, nlevels=20,
	asp=1
#	color.palette=colorRampPalette( c("blue", "white", "red") )
)

top_area_x <- c(left_top_x*0.99, zeta_slice$x, right_top_x*1.01)
top_area_z <- c(left_top_z*1.10, zeta_slice$z, right_top_z*1.10)
#polygon(top_area_x, top_area_z, lwd=4, border=NA, col='white')
lines(zeta_slice$x, zeta_slice$z, lwd=4)
#image(x, z, u, c(-2,2), col=heat.colors(128))
box()
