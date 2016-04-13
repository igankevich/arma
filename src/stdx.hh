#ifndef STDX_HH
#define STDX_HH

namespace stdx {

	template<class T>
	bool
	isnan(T rhs) noexcept {
		return std::isnan(rhs);
	}

}

#endif // STDX_HH
