#ifndef VELOCITY_POTENTIAL_FIELD_HH
#define VELOCITY_POTENTIAL_FIELD_HH

#include "types.hh"
#include "domain.hh"

namespace arma {

	template<class T>
	class Velocity_potential_field {

	protected:
		Vec2<T> _wnmax;
		T _depth;
		Domain2<T> _domain;

		virtual Array2D<T>
		compute_velocity_field_2d(
			Array3D<T>& zeta,
			const size2 arr_size,
			const T z,
			const int idx_t
		) = 0;

		virtual void
		write(std::ostream& out) const {
			out << "wnmax=" << _wnmax << ','
				<< "depth=" << _depth << ','
				<< "domain=" << _domain;
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
		operator()(Array3D<T>& zeta, const Domain3D& subdomain) {
			using blitz::Range;
			const size3 zeta_size = subdomain.ubound() - subdomain.lbound();
			const size2 arr_size(zeta_size(1), zeta_size(2));
			const int nt = _domain.num_points(0);
			const int nz = _domain.num_points(1);
			Array4D<T> result(blitz::shape(
				nt, nz,
				zeta_size(1), zeta_size(2)
			));
			#if ARMA_OPENMP
			#pragma omp parallel for collapse(2)
			#endif
			for (int i=0; i<nt; ++i) {
				for (int j=0; j<nz; ++j) {
					const Vec2<T> p = _domain({i,j});
					const T t = p(0);
					const T z = p(1);
					result(i, j, Range::all(), Range::all()) =
					compute_velocity_field_2d(
						zeta, arr_size,
						z, t
					);
				}
//				std::clog << "Finished time slice ["
//					<< (i+1) << '/' << nt << ']'
//					<< std::endl;
			}
			return result;
		}

		Array4D<T>
		operator()(Array3D<T>& zeta) {
			return operator()(zeta, zeta.domain());
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
			sys::parameter_map params({
			    {"wnmax", sys::make_param(rhs._wnmax, validate_finite<T,2>)},
			    {"depth", sys::make_param(rhs._depth, validate_finite<T>)},
			    {"domain", sys::make_param(rhs._domain, validate_domain<T,2>)},
			}, true);
			in >> params;
			return in;
		}

	};

}

#endif // VELOCITY_POTENTIAL_FIELD_HH
