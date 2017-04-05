Name:           arma
Version:        0.9
Release:        1%{?dist}
Summary:        ARMA ocean wavy surface simulation programme.

License:        GPLv2
URL:            http://igankevich.com/
Source0:        arma-%{version}.tar.gz
BuildRoot: %{_tmppath}/%{name}-%{version}-%{release}-root-%(%{__id_u} -n)

%description
ARMA ocean wavy surface simulation programme.


%package        devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}

%description    devel
The %{name}-devel package contains libraries and header files for
developing applications that use %{name}.


%global debug_package %{nil}


%prep
%autosetup


%build
meson --prefix %{_prefix} -Dframework=openmp . build


%install
DESTDIR=%{buildroot} ninja-build -C build install

%check
ninja-build -C build test


%post -p /sbin/ldconfig

%postun -p /sbin/ldconfig


%files
%{_bindir}/arma
%{_bindir}/arma-dcmt
%{_bindir}/arma-visual
%{_bindir}/arma-realtime

%files devel

%changelog
