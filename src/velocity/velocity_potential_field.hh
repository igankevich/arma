#ifndef VELOCITY_POTENTIAL_FIELD_HH
#define VELOCITY_POTENTIAL_FIELD_HH

#if ARMA_OPENMP
#include <omp.h>
#endif
#include "types.hh"
#include "domain.hh"

namespace arma {

	template<class T>
	class Velocity_potential_field {

	protected:
		Vec2D<T> _wnmax;
		T _depth;
		Domain2<T> _domain;

		virtual void
		precompute(const Array3D<T>& zeta) {}

		virtual void
		precompute(const Array3D<T>& zeta, const int idx_t) {}

		virtual Array2D<T>
		compute_velocity_field_2d(
			const Array3D<T>& zeta,
			const Shape2D arr_size,
			const T z,
			const int idx_t
		) = 0;

		virtual void
		write(std::ostream& out) const {
			out << "wnmax=" << _wnmax << ','
				<< "depth=" << _depth << ','
				<< "domain=" << _domain;
		}

		virtual void
		read(std::istream& in) {
			sys::parameter_map params({
			    {"wnmax", sys::make_param(_wnmax, validate_finite<T,2>)},
			    {"depth", sys::make_param(_depth, validate_finite<T>)},
			    {"domain", sys::make_param(_domain, validate_domain<T,2>)},
			}, true);
			in >> params;
		}

	public:
		Velocity_potential_field() = default;
		Velocity_potential_field(const Velocity_potential_field&) = default;
		Velocity_potential_field(Velocity_potential_field&&) = default;
		virtual ~Velocity_potential_field() = default;

		/**
		\param[in] zeta      ocean wavy surface
		\param[in] subdomain region of zeta
		\param[in] z         a coordinate \f$z\f$ in which to compute velocity
		                     potential
		\param[in] idx_t     a time point in which to compute velocity potential,
		                     specified as index of zeta
		*/
		Array4D<T>
		operator()(const Array3D<T>& zeta) {
			using blitz::Range;
			const Shape3D& zeta_size = zeta.shape();
			const Shape2D arr_size(zeta_size(1), zeta_size(2));
			const int nt = _domain.num_points(0);
			const int nz = _domain.num_points(1);
			Array4D<T> result(blitz::shape(
				nt, nz,
				zeta_size(1), zeta_size(2)
			));
			precompute(zeta);
			for (int i=0; i<nt; ++i) {
				const T t = _domain(i, 0);
				precompute(zeta, t);
				#if ARMA_OPENMP
				#pragma omp parallel for
				#endif
				for (int j=0; j<nz; ++j) {
					const T z = _domain(j, 1);
					result(i, j, Range::all(), Range::all()) =
					compute_velocity_field_2d(zeta, arr_size, z, t);
				}
//				std::clog << "Finished time slice ["
//					<< (i+1) << '/' << nt << ']'
//					<< std::endl;
			}
			return result;
		}

		const Domain2<T>
		domain() const noexcept {
			return _domain;
		}

		friend std::ostream&
		operator<<(std::ostream& out, const Velocity_potential_field& rhs) {
			rhs.write(out);
			return out;
		}

		friend std::istream&
		operator>>(std::istream& in, Velocity_potential_field& rhs) {
			rhs.read(in);
			return in;
		}

	};

}

#endif // VELOCITY_POTENTIAL_FIELD_HH
