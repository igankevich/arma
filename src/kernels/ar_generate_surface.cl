#include "common.h"

#define ZETA_IDX(i,j,k) (((i)*z2 + (j))*z3 + (k))
#define PHI_IDX(l,m,n) (((l)*p2 + (m))*p3 + (n))

kernel void
ar_generate_surface(
	global const T* phi,
	const Shape3D phi_shape,
	global T* zeta,
	const Shape3D zeta_shape,
	const Shape3D zeta_lower,
	const Shape3D zeta_upper
) {
	// phi shape
	const int p1 = phi_shape.s0;
	const int p2 = phi_shape.s1;
	const int p3 = phi_shape.s2;
	// zeta shape
	const int z2 = zeta_shape.s1;
	const int z3 = zeta_shape.s2;
	// zeta lower boundary
	const int i0 = zeta_lower.s0;
	const int j0 = zeta_lower.s1;
	const int k0 = zeta_lower.s2;
	// zeta upper boundary
	const int i1 = zeta_upper.s0;
	const int j1 = zeta_upper.s1;
	const int k1 = zeta_upper.s2;
	for (int i=i0; i<=i1; ++i) {
		for (int j=j0; j<=j1; ++j) {
			for (int k=k0; k<=k1; ++k) {
				const int m1 = min(i+1, p1);
				const int m2 = min(j+1, p2);
				const int m3 = min(k+1, p3);
				T sum = 0;
				for (int l=0; l<m1; ++l) {
					for (int m=0; m<m2; ++m) {
						for (int n=0; n<m3; ++n) {
							sum += phi[PHI_IDX(l,m,n)]
								* zeta[ZETA_IDX(i-l,j-m,k-n)];
						}
					}
				}
				zeta[ZETA_IDX(i,j,k)] += sum;
			}
		}
	}
}


