#include "basic_model.hh"

#include "config.hh"

#include "bits/bscheduler_io.hh"

#if ARMA_BSCHEDULER

template <class T>
void
arma::generator::Basic_model<T>
::write(sys::pstream& out) const {
	bsc::kernel::write(out);
	out << this->_outgrid;
	out << this->_noseed;
	out << this->_zeta;
	out << this->_mts;
}

template <class T>
void
arma::generator::Basic_model<T>
::read(sys::pstream& in) {
	bsc::kernel::read(in);
	in >> this->_outgrid;
	in >> this->_noseed;
	in >> this->_zeta;
	in >> this->_mts;
}

#endif

template class arma::generator::Basic_model<ARMA_REAL_TYPE>;
