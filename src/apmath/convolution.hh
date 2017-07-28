#ifndef APMATH_CONVOLUTION_HH
#define APMATH_CONVOLUTION_HH

#include "fourier.hh"
#include "blitz.hh"
#include <stdexcept>
#include <vector>
#include <mutex>
#include <iostream>
#include "physical_constants.hh"

namespace arma {

	namespace apmath {

		/*
		\brief Multidimensional convolution based on Fourier transform.

		Slicing is done in specified dimension with specified padding.
		*/
		template <class T, int N>
		class Convolution {

		public:
			typedef Fourier_transform<T,N> transform_type;
			typedef typename transform_type::shape_type shape_type;
			typedef typename transform_type::array_type array_type;
			typedef blitz::RectDomain<N> domain_type;

		private:
			transform_type _fft;
			int _dimension;
			int _blocksize;
			int _padding;

		public:
			inline explicit
			Convolution(
				const shape_type& kernel_shape,
				int dimension,
				int blocksize,
				int padding
			):
			_fft(zero_pad_shape(kernel_shape, blocksize, dimension, padding)),
			_dimension(dimension),
			_blocksize(blocksize),
			_padding(padding)
			{ check(); }

			inline array_type
			convolve(array_type signal, array_type kernel) {
				using blitz::all;
				using std::min;
				using constants::_2pi;
				/// Zero-pad kernel to be of length `block_size + padding`.
				const shape_type padded_block = zero_pad_shape(kernel.shape());
				if (!all(padded_block == this->_fft.shape())) {
					throw std::length_error("bad kernel size");
				}
				domain_type orig_domain(shape_type(0), kernel.shape()-1);
				array_type padded_kernel(padded_block);
				padded_kernel(orig_domain) = kernel;
				const T nelements = padded_kernel.numElements();
				/// Take forward FFT of padded kernel.
				padded_kernel = this->_fft.forward(padded_kernel);
				/// Decompose input signal into blocks of length `block_size`.
				const int bs = this->_blocksize;
				const int pad = this->_padding;
				const int dim = this->_dimension;
				const int limit = signal.extent(dim);
				const int nparts = limit / bs + ((limit%bs == 0) ? 0 : 1);
				if (bs+pad > limit) {
					throw std::length_error("too large block+padding size");
				}
				array_type out_signal(signal.shape());
				std::vector<std::mutex> mutexes(nparts);
				//#if ARMA_OPENMP
				//#pragma omp parallel for schedule(static,1)
				//#endif
				for (int i=0; i<nparts; ++i) {
					/// Zero-pad each part to be of length
					/// `block_size + padding`.
					const int offset = i*bs;
					shape_type from(0);
					from(dim) = offset;
					shape_type to(signal.shape()-1);
					to(dim) = min(limit, offset+bs) - 1;
					domain_type part_domain(from, to);
					domain_type dom_to(from, to);
					dom_to.lbound()(dim) -= offset;
					dom_to.ubound()(dim) -= offset;
					#ifndef NDEBUG
					std::clog << "copy from signal "
						<< part_domain
						<< " to part "
						<< dom_to
						<< std::endl;
					#endif
					array_type padded_part(padded_block);
					padded_part(dom_to) = signal(part_domain);
					/// Take forward FFT of each padded part.
					padded_part = this->_fft.forward(padded_part);
					/// Multiply two FFTs.
					padded_part *= padded_kernel;
					/// Take backward FFT of the result.
					padded_part = this->_fft.backward(padded_part);
					padded_part /= nelements;
					//padded_part.transposeSelf(blitz::secondDim, blitz::firstDim);
					domain_type padded_from(from, to);
					/// Copy padded part back overlapping it with adjacent parts.
					const int tmp = padded_from.ubound()(dim);
					padded_from.ubound()(dim) = min(tmp+pad, limit-1);
					domain_type padded_to(padded_from);
					padded_to.lbound()(dim) -= offset;
					padded_to.ubound()(dim) -= offset;
					#ifndef NDEBUG
					std::clog << "copy from part "
						<< padded_to
						<< " to signal "
						<< padded_from
						<< std::endl;
					#endif
					{
						const int m = (i%2==0) ? i : (i-1);
						std::unique_lock<std::mutex> lock(mutexes[m]);
						out_signal(padded_from) += padded_part(padded_to);
					}
				}
				return out_signal;
			}

		private:

			static inline shape_type
			zero_pad_shape(const shape_type& rhs, int bs, int dim, int padding) {
				shape_type shp(rhs);
				shp(dim) = bs + padding;
				return shp;
			}

			inline shape_type
			zero_pad_shape(const shape_type& rhs) {
				return zero_pad_shape(
					rhs,
					this->_blocksize,
					this->_dimension,
					this->_padding
				);
			}

			inline void
			check() {
				if (this->_dimension < 0 || this->_dimension >= N) {
					throw std::out_of_range("dimension out of range");
				}
				if (this->_padding < 0) {
					throw std::length_error("bad padding");
				}
				if (this->_blocksize <= 0) {
					throw std::length_error("bad block size");
				}
			}

		};

	}

}

#endif // APMATH_CONVOLUTION_HH
