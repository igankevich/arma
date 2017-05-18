#ifndef KERNELS_HARTS_H
#define KERNELS_HARTS_H

int
rotate_right(const int idx, const int n) {
	return (idx + n/2)%n;
}

#endif // KERNELS_HARTS_H
