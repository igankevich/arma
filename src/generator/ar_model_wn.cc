namespace {

	template <class T, class Generator>
	arma::Array3D<T>
	generate_white_noise(
		const arma::Shape3D& size,
		T variance,
		Generator generator
	) {
		std::normal_distribution<T> normal(T(0), std::sqrt(variance));
		arma::Array3D<T> eps(size);
		std::generate_n(
			eps.data(),
			eps.numElements(),
			std::bind(normal, generator)
		);
		return eps;
	}

}
