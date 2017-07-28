#ifndef APMATH_CONVOLUTION_HH
#define APMATH_CONVOLUTION_HH

#include "fourier.hh"
#include "blitz.hh"
#include <stdexcept>
#include <mutex>
#include <iostream>

namespace arma {

	namespace bits {

		template <class T, int N>
		class Index {

		public:
			typedef blitz::TinyVector<T,N> shape_type;

		private:
			shape_type _shape;

		public:
			inline explicit
			Index(const shape_type& shape):
			_shape(shape)
			{}

			inline shape_type
			operator()(int linear_index) const noexcept {
				shape_type idx;
				for (int i=0; i<N; ++i) {
					int res = linear_index;
					for (int j=N-1; j>i; --j) {
						res /= this->_shape(j);
					}
					idx(i) = res % this->_shape(i);
				}
				return idx;
			}

			const int
			num_elements() const noexcept {
				return blitz::product(this->_shape);
			}

		};

	}

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
			shape_type _blocksize;
			shape_type _padding;
			transform_type _fft;

		public:
			inline explicit
			Convolution(
				const shape_type& blocksize,
				const shape_type& padding
			):
			_blocksize(blocksize),
			_padding(padding),
			_fft(blocksize + padding)
			{ check(); }

			inline array_type
			convolve(array_type signal, array_type kernel) {
				using blitz::all;
				using blitz::product;
				using blitz::div_ceil;
				using blitz::min;
				if (!all(kernel.shape() <= this->_blocksize)) {
					throw std::length_error("bad kernel shape");
				}
				/// Zero-pad kernel to be of length `block_size + padding`.
				const shape_type padded_block = this->_blocksize + this->_padding;
				domain_type orig_domain(shape_type(0), kernel.shape()-1);
				array_type padded_kernel(padded_block);
				padded_kernel(orig_domain) = kernel;
				const T nelements = padded_kernel.numElements();
				/// Take forward FFT of padded kernel.
				padded_kernel = this->_fft.forward(padded_kernel);
				/// Decompose input signal into blocks of length `block_size`.
				const shape_type bs = this->_blocksize;
				const shape_type pad = this->_padding;
				const shape_type limit = signal.shape();
				const shape_type nparts = div_ceil(limit, bs);
				const bits::Index<int,N> part_index(nparts);
				const int all_parts = part_index.num_elements();
				if (!all(bs <= limit)) {
					throw std::length_error("bad block size");
				}
				array_type out_signal(limit);
				blitz::Array<std::mutex,N> mutexes(nparts);
				//#if ARMA_OPENMP
				//#pragma omp parallel for collapse(2) schedule(static,1) ordered
				//#endif
				for (int i=0; i<all_parts; ++i) {
					/// Zero-pad each part to be of length
					/// `block_size + padding`.
					const shape_type idx = part_index(i);
					const shape_type offset = idx*bs;
					shape_type from = offset;
					shape_type to = min(limit, offset+bs) - 1;
					domain_type part_domain(from, to);
					domain_type dom_to(from-offset, to-offset);
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
					/// Copy padded part back overlapping it with adjacent parts.
					domain_type padded_from(from, min(to+pad, limit-1));
					domain_type padded_to(from-offset, padded_from.ubound()-offset);
					#ifndef NDEBUG
					std::clog << "copy from part "
						<< padded_to
						<< " to signal "
						<< padded_from
						<< std::endl;
					#endif
					//#if ARMA_OPENMP
					//#pragma omp ordered
					//#endif
					{
						out_signal(padded_from) += padded_part(padded_to);
					}
				}
				return out_signal;
			}

		private:

			inline void
			check() {
				using blitz::all;
				if (!all(this->_padding >= 0)) {
					throw std::length_error("bad padding");
				}
				if (!all(this->_blocksize > 0)) {
					throw std::length_error("bad block size");
				}
				if (!all(this->_blocksize >= this->_padding)) {
					throw std::length_error("bad block size/padding ratio");
				}
			}

		};

	}

}

#endif // APMATH_CONVOLUTION_HH
