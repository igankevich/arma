namespace {

	template <class T>
	void
	ar_generate_surface(
		arma::Array3D<T>& zeta,
		arma::Array3D<T> phi,
		const arma::Domain3D& subdomain
	) {
		using namespace arma;
		const Shape3D fsize = phi.shape();
		const Shape3D& lbound = subdomain.lbound();
		const Shape3D& ubound = subdomain.ubound();
		const int t0 = lbound(0);
		const int x0 = lbound(1);
		const int y0 = lbound(2);
		const int t1 = ubound(0);
		const int x1 = ubound(1);
		const int y1 = ubound(2);
		for (int t = t0; t <= t1; t++) {
			for (int x = x0; x <= x1; x++) {
				for (int y = y0; y <= y1; y++) {
					const int m1 = std::min(t + 1, fsize[0]);
					const int m2 = std::min(x + 1, fsize[1]);
					const int m3 = std::min(y + 1, fsize[2]);
					T sum = 0;
					for (int k = 0; k < m1; k++)
						for (int i = 0; i < m2; i++)
							for (int j = 0; j < m3; j++)
								sum += phi(k, i, j) *
								       zeta(t - k, x - i, y - j);
					zeta(t, x, y) += sum;
				}
			}
		}
	}

}
