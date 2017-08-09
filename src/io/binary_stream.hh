#ifndef IO_BINARY_STREAM_HH
#define IO_BINARY_STREAM_HH

#include <streambuf>
#include <fstream>
#include <string>
#include "types.hh"
#include "bits/byte_swap.hh"

namespace arma {

	namespace io {

		class Binary_stream {

		private:
			std::streambuf* _buffer = nullptr;

		public:

			inline explicit
			Binary_stream(std::streambuf* buffer) noexcept:
			_buffer(buffer)
			{}

			inline explicit
			Binary_stream(const std::string& filename) noexcept {
				std::filebuf* buf = new std::filebuf;
				buf->open(filename, std::ios::out | std::ios::binary);
				this->_buffer = buf;
			}

			inline
			~Binary_stream() {
				this->close();
			}

			template <class T>
			void
			write(const Array3D<T>& rhs, int t0, int n) {
				const T* first = &rhs(t0,0,0);
				const T* last = first + n*rhs.extent(1)*rhs.extent(2);
				while (first != last) {
					Bytes<T> bytes(*first);
					bytes.to_network_format();
					this->_buffer->sputn(bytes.begin(), bytes.size());
					++first;
				}
			}

			template <class T>
			void
			write(const Array3D<T>& rhs) {
				this->write(rhs, 0, rhs.extent(0));
			}

			inline void
			close() {
				if (this->_buffer) {
					this->_buffer->pubsync();
				}
				delete this->_buffer;
				this->_buffer = nullptr;
			}

		};

	}

}

#endif // IO_BINARY_STREAM_HH
