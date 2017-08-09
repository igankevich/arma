#ifndef APMATH_CONVOLUTION_HH
#define APMATH_CONVOLUTION_HH

#include "fourier.hh"
#include "blitz.hh"
#include "bits/index.hh"
#include <stdexcept>
#include <iostream>
#if ARMA_OPENMP
#include <omp.h>
#endif

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
			typedef Fourier_workspace<T,N> workspace_type;

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

			inline explicit
			Convolution(
				const array_type& signal,
				const array_type& kernel
			):
			_blocksize(get_block_shape(signal.shape(), kernel.shape())),
			_padding(kernel.shape()),
			_fft(_blocksize + _padding)
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
				std::clog << "all_parts=" << all_parts << std::endl;
				array_type out_signal(limit);
				#if ARMA_OPENMP
				#pragma omp parallel
				#endif
				{
					// per-thread workspace
					workspace_type workspace(padded_block);
					#if ARMA_OPENMP
					#pragma omp for schedule(static,1)
					#endif
					for (int i=0; i<all_parts; ++i) {
						/// Zero-pad each part to be of length
						/// `block_size + padding`.
						const shape_type idx = part_index(i);
						const shape_type offset = idx*bs;
						const shape_type from = offset;
						const shape_type to = min(limit, offset+bs) - 1;
						const domain_type part_domain(from, to);
						const domain_type dom_to(from-offset, to-offset);
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
						padded_part = this->_fft.forward(padded_part, workspace);
						/// Multiply two FFTs.
						padded_part *= padded_kernel;
						/// Take backward FFT of the result.
						padded_part = this->_fft.backward(padded_part, workspace);
						padded_part /= nelements;
						/// Copy padded part back overlapping it with adjacent parts.
						const domain_type padded_from(from, min(to+pad, limit-1));
						const domain_type padded_to(from-offset, padded_from.ubound()-offset);
						#ifndef NDEBUG
						std::clog << "copy from part "
							<< padded_to
							<< " to signal "
							<< padded_from
							<< std::endl;
						#endif
						#if ARMA_OPENMP
						#pragma omp critical
						#endif
						{
							out_signal(padded_from) += padded_part(padded_to);
						}
					}
				}
				return out_signal;
			}

		private:

			inline void
			check() {
				std::clog << "this->_blocksize=" << this->_blocksize << std::endl;
				std::clog << "this->_padding=" << this->_padding << std::endl;
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

			inline shape_type
			get_block_shape(
				const shape_type& signal_shape,
				const shape_type& kernel_shape
			) {
				using blitz::min;
				using blitz::abs;
				#if ARMA_OPENMP
				const int parallelism = omp_get_max_threads();
				#else
				const int parallelism = 2;
				#endif
				shape_type guess1 = min(signal_shape, 4*kernel_shape);
				shape_type guess2 = max(kernel_shape, signal_shape / parallelism);
				return min(guess1, guess2) + abs(guess1 - guess2) / 2;
			}

		};

	}

}

#endif // APMATH_CONVOLUTION_HH
