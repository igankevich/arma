#ifndef BITS_INDEX_HH
#define BITS_INDEX_HH

#include <blitz/array.h>

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

			int
			num_elements() const noexcept {
				return blitz::product(this->_shape);
			}

		};

	}

}

#endif // BITS_INDEX_HH
