#include <gtest/gtest.h>

#include <cmath>

#include "derivative.hh"
#include "discrete_function.hh"
#include "opencl/array.hh"
#include "opencl/cl.hh"
#include "opencl/device_type.hh"
#include "opencl/opencl.hh"
#include "opencl/vec.hh"
#include "opengl.hh"
#include "profile.hh"
#include "velocity/high_amplitude_realtime_solver.hh"

using namespace arma;

typedef ARMA_REAL_TYPE T;

arma::Array3D<T> fill_sin(int nx, int ny, int nz)
{
    arma::Array3D<T> zeta(blitz::shape(nx, ny, nz));
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                zeta(i, j, k) = sin(T(i + j + k) / T(100));
    return zeta;
}

TEST(Derivatives, Accuracy)
{
    const int nt = 10;
    const int nx = 10;
    const int ny = 10;
    arma::Array3D<T> zeta = fill_sin(nt, nx, ny);
    zeta.copy_to_device();
    arma::Array3D<T> dzeta(blitz::shape(nt, nx, ny));
    dzeta = 0;
    dzeta.copy_to_device();
    size_t max_memory = opencl::devices()[0].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    cl::Kernel kernel = opencl::get_kernel("compute_second_function");
    T delta = T(0.01);

    kernel.setArg(0, zeta.buffer());
    kernel.setArg(1, delta);
    kernel.setArg(2, 0); // t derivative
    kernel.setArg(3, dzeta.buffer());
    kernel.setArg(4, max_memory, NULL);

    zeta.compute(kernel);

    dzeta.copy_to_host();

    T max_residual = 0;

    for (int i = 0; i < nt; i++)
    {
        using blitz::abs;
        using blitz::max;
        using blitz::min;
        using blitz::Range;
        Array2D<T> tmp(dzeta(i, Range::all(), Range::all()));
        Array2D<T> real_derivative =
            derivative<0, T>(zeta, arma::Vec3D<T>(0.01, 0.01, 0.01), i);
        T residual = min(abs(real_derivative - dzeta(i, Range::all(), Range::all())));
        max_residual = std::max(residual, max_residual);
    }
    EXPECT_NEAR(max_residual, T(0), T(1e-3));
}