#ifndef AUTOREG_HH
#define AUTOREG_HH

/// @file
/// File with subroutines for AR model, Yule-Walker equations
/// and some others.

namespace autoreg {

template<class T>
struct Variance_WN {
	Variance_WN(const std::valarray<T>& ar_coefs_, const std::valarray<T>& acf_):
	ar_coefs(ar_coefs_), acf(acf_), sum(0) {}

	void act() {
		const int n = ar_coefs.size();
		for (int i=0; i<n; ++i) {
			sum += ar_coefs[i]*acf[i];
		}
		sum = acf[0] - sum;
	}

	T getsum() const {
		return sum;
	}

private:
	const std::valarray<T>& ar_coefs;
	const std::valarray<T>& acf;
	T sum;
};

template<class T> T
var_acf(std::valarray<T>& acf) {
	return acf[0];
}

template<class T>
struct Yule_walker {

	Yule_walker(const std::valarray<T>& acf_, const size3& acf_size_, std::valarray<T>& a_, std::valarray<T>& b_):
	acf(acf_), acf_size(acf_size_), a(a_), b(b_)
	{}

	void act() {
		const int n = acf.size() - 1;
		for (int i=0; i<n; ++i) {
			const int n = acf.size()-1;
			const Index<3> id(acf_size);
			const Index<2> ida(size2(n, n));
			for (int j=0; j<n; j++) {
			    // casting to signed type ptrdiff_t
				int i2 = id(sub_abs(id.x(i+1), id.x(j+1)),
				            sub_abs(id.y(i+1), id.y(j+1)),
				            sub_abs(id.t(i+1), id.t(j+1)) );
				int i1 = i*n + j; //ida(i, j);
//				cout << "i  = " << i << endl;
//				cout << "j  = " << j << endl;
//				cout << "i2 = " << i2 << endl;
				a[i1] = acf[i2];
			}
		}
		const int m = b.size();
		for (int i=0; i<m; ++i) {
			const Index<3> id(acf_size);
			b[i] = acf[id( id.x(i+1), id.y(i+1), id.t(i+1) )];
		}
	}

private:
	int sub_abs(int a, int b) {
	    return (a > b) ? a-b : b-a;
	}

	const std::valarray<T>& acf;
	const size3& acf_size;
	std::valarray<T>& a;
	std::valarray<T>& b;
};

template<class T>
bool is_stationary(std::valarray<T>& ar_coefs) {
	int n = ar_coefs.size();
	for (int i=0; i<n; ++i)
		if (std::abs(ar_coefs[i]) > T(1))
			return false;
	return true;
}


template<class T>
void approx_acf(const T alpha,
                const T beta,
				const T gamm,
				const Vector<T, 3>& delta,
				const size3& acf_size,
				std::valarray<T>& acf)
{
	const Index<3> id(acf_size);
    for (std::size_t t=0; t<acf_size[0]; t++) {
        for (std::size_t x=0; x<acf_size[1]; x++) {
            for (std::size_t y=0; y<acf_size[2]; y++) {
				const T k1 = t*delta[0] + x*delta[1] + y*delta[2];
				acf[id(t, x, y)] = gamm*exp(-alpha*k1)*cos(beta*t*delta[0])*cos(beta*x*delta[1]);//*cos(beta*y*delta[2]);
			}
		}
	}
}

template<class T>
struct Solve_Yule_Walker {

	Solve_Yule_Walker(std::valarray<T>& ar_coefs2, std::valarray<T>& aa, std::valarray<T>& bb, const size3& acf_size):
		ar_coefs(ar_coefs2), a(aa), b(bb), _acf_size(acf_size)
	{}

	void act() {

		int m = ar_coefs.size()-1;
		int info = 0;
		sysv<T>('U', m, 1, &a[0], m, &b[0], m, &info);
		if (info != 0) {
			std::stringstream s;
			s << "ssysv error, D(" << info << ", " << info << ")=0";
			throw std::invalid_argument(s.str());
		}

		std::copy(&b[0], &b[m], &ar_coefs[1]);
		ar_coefs[0] = 0;

		if (!is_stationary(ar_coefs)) {
			std::stringstream msg;
			msg << "Process is not stationary: |f(i)| > 1\n";
//			int n = ar_coefs.size();
			Index<3> idx(_acf_size);
			for (size_t i=0; i<_acf_size[0]; ++i)
				for (size_t j=0; j<_acf_size[1]; ++j)
					for (size_t k=0; k<_acf_size[2]; ++k)
						if (std::abs(ar_coefs[idx(i, j, k)]) > T(1))
							msg << "ar_coefs[" << i << ',' << j << ',' << k << "] = " << ar_coefs[idx(i, j, k)] << '\n';
			throw std::runtime_error(msg.str());
//			std::cerr << "Continue anyway? y/N\n";
//			char answer = 'n';
//			std::cin >> answer;
//			if (answer == 'n' || answer == 'N') {
//				throw std::runtime_error("Process is not stationary: |f[i]| >= 1.");
//			}
		}
	}

private:
	std::valarray<T>& ar_coefs;
	std::valarray<T>& a;
	std::valarray<T>& b;
	const size3& _acf_size;
};

template<class T>
struct Autoreg_coefs {
	Autoreg_coefs(const std::valarray<T>& acf_model_,
				   const size3& acf_size_,
				   std::valarray<T>& ar_coefs_):
		acf_model(acf_model_), acf_size(acf_size_), ar_coefs(ar_coefs_),
		a((ar_coefs.size()-1)*(ar_coefs.size()-1)), b(ar_coefs.size()-1)
	{}

	void act() {
		Yule_walker<T> generate_yule_walker_equations(acf_model, acf_size, a, b);
		generate_yule_walker_equations.act();
		Solve_Yule_Walker<T> solve_yule_walker_equations(ar_coefs, a, b, acf_size);
		solve_yule_walker_equations.act();
	}

private:
	const std::valarray<T>& acf_model;
	const size3& acf_size;
	std::valarray<T>& ar_coefs;
	std::valarray<T> a;
	std::valarray<T> b;
};

/// Генерация белого шума по алгоритму Вихря Мерсенна и
/// преобразование его к нормальному распределению по алгоритму Бокса-Мюллера.
template<class T>
void generate_white_noise(std::valarray<T>& eps, const size3& zsize, const T var_eps)
{
    mt_struct_stripped d_MT[MT_RNG_COUNT];
	read_mt_params(d_MT, "MersenneTwister.dat");

	if (var_eps < T(0)) {
		std::stringstream msg;
		msg << "Variance of white noise is lesser than zero: " << var_eps;
		throw std::runtime_error(msg.str());
	}

    const T sqrtVarA = sqrt(var_eps);
    const std::size_t part = 0;
    const std::size_t t0 = 0;
    const std::size_t t1 = zsize[0];
    const Index<3> idz(zsize);

//	clog << "Part = " << part << endl
//	     << "t0 = " << t0 << endl
//		 << "t1 = " << t1 << endl;

    int iState, iState1, iStateM;
    unsigned int mti, mti1, mtiM, x;
    unsigned int mt[MT_NN], matrix_a, mask_b, mask_c;

    // load bit-vector Mersenne Twister parameters
    matrix_a = d_MT[part].matrix_a;
    mask_b   = d_MT[part].mask_b;
    mask_c   = d_MT[part].mask_c;

    // initialize current state
    mt[0] = d_MT[part].seed;
    for (iState = 1; iState < MT_NN; iState++)
        mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

    iState = 0;
    mti1 = mt[0];

    const std::size_t ni = zsize[1];
    const std::size_t nj = zsize[2];

    for (std::size_t k=t0; k<t1; k++) {
        for (std::size_t i=0; i<ni; i++) {
            for (std::size_t j=0; j<nj; j++) {
                iState1 = iState + 1;
                iStateM = iState + MT_MM;
                if (iState1 >= MT_NN) iState1 -= MT_NN;
                if (iStateM >= MT_NN) iStateM -= MT_NN;
                mti  = mti1;
                mti1 = mt[iState1];
                mtiM = mt[iStateM];

                // MT recurrence
                x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
                x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

                mt[iState] = x;
                iState = iState1;

                //Tempering transformation
                x ^= (x >> MT_SHIFT0);
                x ^= (x << MT_SHIFTB) & mask_b;
                x ^= (x << MT_SHIFTC) & mask_c;
                x ^= (x >> MT_SHIFT1);

                //Convert to (0, 1] float and write to global memory
                eps[idz(k, i, j)] = (x + 1.0f) / 4294967296.0f;
            }
        }
    }

    // Box-Muller transformation
    const std::size_t nk = (t1-t0);
    const std::size_t total = ni*nj*nk;
    const std::size_t half = total/2;
    T eps_saved = 0;
    if (total%2 != 0) {
        // save second element from the end if total count is odd
        std::size_t i = idz(t1, 0, 0);
        eps_saved = eps[i-2];
    }
    for (std::size_t i=0; i<half; i++) {
        std::size_t t = idz(t0, 0, 0) + 2*i;
        std::size_t i1 = t;
        std::size_t i2 = t+1;
        T   r = sqrt(T(-2)*log(eps[i1]))*sqrtVarA;
        T phi = T(2)*M_PI*eps[i2];
        eps[i1] = r*cos(phi);
        eps[i2] = r*sin(phi);
    }
    if (total%2 != 0) {
        std::size_t i2 = idz(t1, 0, 0)-1;
        T   r = sqrt(T(-2)*log(eps_saved))*sqrtVarA;
        T phi = T(2)*M_PI*eps[i2];
        eps[i2] = r*sin(phi);
    }

	// debug NaNs caused by race condition.
    for (std::size_t k=t0; k<t1; k++) {
        for (std::size_t i=0; i<ni; i++) {
            for (std::size_t j=0; j<nj; j++) {
                if (isnan(eps[idz(k, i, j)])) {
                    std::cout << "Nan at box-muller[" << size3(k, i, j) << "]\n";
					exit(1);
                }
            }
        }
    }
}

/// Генерация отдельных частей реализации волновой поверхности.
template<class T>
void generate_zeta(const std::valarray<T>& phi,
				   const size3& fsize,
				   const size3& zsize,
				   std::valarray<T>& zeta)
{
	const Index<3> id(fsize);
	const Index<3> idz(zsize);
	const std::size_t t1 = zsize[0];
	const std::size_t x1 = zsize[1];
	const std::size_t y1 = zsize[2];
    for (std::size_t t=0; t<t1; t++) {
        for (std::size_t x=0; x<x1; x++) {
            for (std::size_t y=0; y<y1; y++) {
                const std::size_t m1 = std::min(t+1, fsize[0]);
                const std::size_t m2 = std::min(x+1, fsize[1]);
                const std::size_t m3 = std::min(y+1, fsize[2]);
                T sum = 0;
                for (std::size_t k=0; k<m1; k++)
                    for (std::size_t i=0; i<m2; i++)
                        for (std::size_t j=0; j<m3; j++)
                            sum += phi[id(k, i, j)]*zeta[idz(t-k, x-i, y-j)];
                zeta[idz(t, x, y)] += sum;
            }
        }
    }
}

/// Удаление участков разгона из реализации.
template<class T>
void trim_zeta(const std::valarray<T>& zeta2,
               const size3& zsize2,
               const size3& zsize,
               std::valarray<T>& zeta)
{
    const Index<3> id1(zsize);
    const Index<3> id2(zsize2);

    const std::size_t t1 = zsize[0];
    const std::size_t x1 = zsize[1];
    const std::size_t y1 = zsize[2];

    const std::size_t dt = zsize2[0] - zsize[0];
    const std::size_t dx = zsize2[1] - zsize[1];
    const std::size_t dy = zsize2[2] - zsize[2];

    for (std::size_t t=0; t<t1; t++) {
        for (std::size_t x=0; x<x1; x++) {
            for (std::size_t y=0; y<y1; y++) {
                const std::size_t x2 = x + dx;
                const std::size_t y2 = y + dy;
                const std::size_t t2 = t + dt;
                zeta[id1(t, x, y)] = zeta2[id2(t2, x2, y2)];
            }
        }
    }
}

template<class T>
struct Wave_surface_generator {

	Wave_surface_generator(const std::valarray<T>& phi_,
						   const size3& fsize_,
						   const T var_eps_,
						   const size3& zsize2_,
						   const size3& zsize_,
						   const Vector<T, 3>& zdelta_):
		phi(phi_), fsize(fsize_), var_eps(var_eps_),
		zsize2(zsize2_), zsize(zsize_), zeta(zsize.count()), zeta2(zsize2.count()), zdelta(zdelta_)
	{}

	void act() {
		generate_white_noise(zeta2, zsize2, var_eps);
		generate_zeta(phi, fsize, zsize2, zeta2);
		trim_zeta(zeta2, zsize2, zsize, zeta);
	}

	const std::valarray<T>& get_wavy_surface() const {
		return zeta;
	}

	std::valarray<T>& get_wavy_surface() {
		return zeta;
	}

private:
	const std::valarray<T>& phi;
	const size3& fsize;
	const T var_eps;
	const size3& zsize2;
	const size3& zsize;
	std::valarray<T> zeta;
	std::valarray<T> zeta2;
	const Vector<T,3> zdelta;
};

}

#endif // AUTOREG_HH
