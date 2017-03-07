#!/usr/bin/Rscript

middle_element <- function (x) {
	x[round(length(x) + 0.5)/2]
}

build_dir <- Sys.getenv('MESON_BUILD_ROOT')
if (!is.null(build_dir) & build_dir != '') {
	setwd(build_dir)
}
phi <- read.csv('phi.csv')
# slice time and Y ranges through the center
slice_t <- middle_element(unique(phi$t))
slice_y <- middle_element(unique(phi$y))
phi_slice <- phi[phi$t == slice_t & phi$y == slice_y & phi$x >= 10 & phi$z >= -2,]
x <- unique(phi_slice$x)
z <- unique(phi_slice$z)
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
zeta_slice <- zeta[zeta$t == slice_t & zeta$y == slice_y & zeta$x >= 10,]
#print(zeta_slice)
#print(x)


# plot the image
require(grDevices)

cairo_pdf(filename="u.pdf",width=10,height=4)
contour(
	x, z, u, nlevels=10,
#	color.palette=colorRampPalette( c("blue", "white", "red") )
)
lines(zeta_slice$x, zeta_slice$z, lwd=4)
#image(x, z, u, c(-2,2), col=heat.colors(128))
box()
