#include "fourier_opencl.hh"

#include "opencl/opencl.hh"
#include "opencl/vec.hh"
#include "util.hh"

#define CHECK(x) ::cl::detail::errHandler((x), #x);

namespace {

	template <int N>
	struct clfft_traits {};

	template <>
	struct clfft_traits<1> {
		static constexpr const clfftDim dim = CLFFT_1D;
	};

	template <>
	struct clfft_traits<2> {
		static constexpr const clfftDim dim = CLFFT_2D;
	};

	template <>
	struct clfft_traits<3> {
		static constexpr const clfftDim dim = CLFFT_3D;
	};

	template <class T>
	constexpr const clfftPrecision clfft_precision =
		std::conditional<
			std::is_same<T,float>::value,
			std::integral_constant<clfftPrecision,CLFFT_SINGLE>,
			std::integral_constant<clfftPrecision,CLFFT_DOUBLE>
		>::type::value;

	void
	debug_print(clfftPlanHandle h) {
		clfftLayout iLayout, oLayout;
		size_t batchSize;
		clfftDim dim;
		cl_uint dimSize;
		size_t iDist, oDist;
		CHECK(clfftGetLayout(h, &iLayout, &oLayout));
		CHECK(clfftGetPlanBatchSize(h, &batchSize));
		CHECK(clfftGetPlanDim(h, &dim, &dimSize));
		CHECK(clfftGetPlanDistance(h, &iDist, &oDist));
		blitz::Array<size_t,1> inStrides(blitz::shape(dimSize));
		CHECK(clfftGetPlanInStride(h, dim, inStrides.data()));
		blitz::Array<size_t,1> outStrides(blitz::shape(dimSize));
		CHECK(clfftGetPlanOutStride(h, dim, outStrides.data()));
		blitz::Array<size_t,1> lengths(blitz::shape(dimSize));
		CHECK(clfftGetPlanLength(h, dim, lengths.data()));
		clfftPrecision precision;
		CHECK(clfftGetPlanPrecision(h, &precision));
		cl_float scale_forward, scale_backward;
		CHECK(clfftGetPlanScale(h, CLFFT_FORWARD, &scale_forward));
		CHECK(clfftGetPlanScale(h, CLFFT_BACKWARD, &scale_backward));
		clfftResultTransposed tr;
		CHECK(clfftGetPlanTransposeResult(h, &tr));
		clfftResultLocation loc;
		CHECK(clfftGetResultLocation(h, &loc));
		size_t tmpbufsize;
		CHECK(clfftGetTmpBufSize(h, &tmpbufsize));
		arma::write_key_value(std::clog, "iLayout", iLayout);
		arma::write_key_value(std::clog, "oLayout", oLayout);
		arma::write_key_value(std::clog, "batchSize", batchSize);
		arma::write_key_value(std::clog, "dim", dim);
		arma::write_key_value(std::clog, "dimSize", dimSize);
		arma::write_key_value(std::clog, "iDist", iDist);
		arma::write_key_value(std::clog, "oDist", oDist);
		arma::write_key_value(std::clog, "inStrides", inStrides);
		arma::write_key_value(std::clog, "outStrides", outStrides);
		arma::write_key_value(std::clog, "lengths", lengths);
		arma::write_key_value(std::clog, "precision", precision);
		arma::write_key_value(std::clog, "scale_forward", scale_forward);
		arma::write_key_value(std::clog, "scale_backward", scale_backward);
		arma::write_key_value(std::clog, "tr", tr);
		arma::write_key_value(std::clog, "loc", loc);
		arma::write_key_value(std::clog, "tmpbufsize", tmpbufsize);
	}

}

template <class T, int N>
arma::apmath::Fourier_transform<T,N>::Fourier_transform() {
	CHECK(clfftInitSetupData(&this->_fft));
	CHECK(clfftSetup(&this->_fft));
}

template <class T, int N>
arma::apmath::Fourier_transform<T,N>::~Fourier_transform() {
	CHECK(clfftDestroyPlan(&this->_fftplan));
}

template <class T, int N>
void
arma::apmath::Fourier_transform<T,N>::init(const shape_type& shape) {
	using blitz::product;
	using opencl::context;
	using opencl::command_queue;
	typedef blitz::TinyVector<size_t,N> clfft_shape;
	const int nelements = product(shape);
	if (nelements <= 0) {
		throw std::invalid_argument("bad shape");
	}
	this->_shape = shape;
	clfft_shape tmp(this->_shape);
	std::clog << "tmp=" << tmp << std::endl;
	std::clog << "context()()=" << context()() << std::endl;
	CHECK(clfftCreateDefaultPlan(
		&this->_fftplan,
		context()(),
		clfft_traits<N>::dim,
		tmp.data()
	));
	CHECK(clfftSetLayout(
		this->_fftplan,
		CLFFT_COMPLEX_INTERLEAVED,
		CLFFT_COMPLEX_INTERLEAVED
	));
	CHECK(clfftSetResultLocation(this->_fftplan, CLFFT_INPLACE));
	CHECK(clfftSetPlanPrecision(this->_fftplan, clfft_precision<T>));
	CHECK(clfftSetPlanScale(this->_fftplan, CLFFT_FORWARD, cl_float(1)));
	CHECK(clfftSetPlanScale(this->_fftplan, CLFFT_BACKWARD, cl_float(1) / nelements));
	CHECK(clfftBakePlan(this->_fftplan, 1, &command_queue()(), nullptr, nullptr));
	debug_print(this->_fftplan);
}

template <class T, int N>
typename arma::apmath::Fourier_transform<T,N>::array_type
arma::apmath::Fourier_transform<T,N>::transform(
	array_type rhs,
	workspace_type& workspace,
	Fourier_direction dir
) {
	using opencl::command_queue;
	using opencl::context;
	using blitz::product;
	cl::Buffer brhs(
		context(),
		CL_MEM_READ_WRITE,
		rhs.numElements()*sizeof(T)
	);
	brhs = this->transform(brhs, workspace, dir);
	cl::copy(
		opencl::command_queue(),
		brhs,
		rhs.data(),
		rhs.data() + rhs.numElements()
	);
	return rhs;
}

template <class T, int N>
typename arma::apmath::Fourier_transform<T,N>::buffer_type
arma::apmath::Fourier_transform<T,N>::transform(
	buffer_type rhs,
	workspace_type& workspace,
	Fourier_direction dir
) {
	using opencl::command_queue;
	using opencl::context;
	using blitz::product;
	CHECK(clfftEnqueueTransform(
		this->_fftplan,
		dir == Fourier_direction::Forward ? CLFFT_FORWARD : CLFFT_BACKWARD,
		1,
		&command_queue()(),
		0,
		nullptr,
		nullptr,
		&rhs(),
		nullptr,
		nullptr
	));
	return rhs;
}

template class arma::apmath::Fourier_transform<std::complex<ARMA_REAL_TYPE>,1>;
template class arma::apmath::Fourier_transform<std::complex<ARMA_REAL_TYPE>,2>;
template class arma::apmath::Fourier_transform<std::complex<ARMA_REAL_TYPE>,3>;
