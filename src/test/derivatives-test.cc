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
                zeta(i, j, k) = sin(T(3 * i +  2 * j + k) / T(100));
    return zeta;
}

TEST(Derivatives, Accuracy)
{
    const int nt = 10;
    const int nx = 17;
    const int ny = 16;
    arma::Array3D<T> zeta = fill_sin(nt, nx, ny);
    zeta.copy_to_device();
    arma::Array3D<T> dzeta(blitz::shape(nt, nx, ny));
    dzeta = 0;
    dzeta.copy_to_device();
    
    int max_memory = opencl::devices()[0].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    int workgroup = opencl::devices()[0].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    // lt*lx*ly <= CL_DEVICE_MAX_WORK_GROUP_SIZE
    // and
    // (lt+8) * lx * ly * sizeof(double) <= CL_DEVICE_LOCAL_MEM_SIZE

    int lx = 8; 
    int ly = 8;
    int lt = std::min(workgroup / lx / ly, (max_memory / lx / ly / (int)sizeof(T) - 8 ));
    int gt = ((nt - 1) / lt + 1) * lt;
    int gx = ((nx - 1) / lx + 1) * lx;
    int gy = ((ny - 1) / ly + 1) * ly;

    ASSERT_TRUE(lt*lx*ly <= workgroup);
    ASSERT_TRUE((lt+8) * lx * ly * (int)sizeof(T) < max_memory);
    ASSERT_TRUE(gt % lt == 0);
    ASSERT_TRUE(gx % lx == 0);
    ASSERT_TRUE(gy % ly == 0);


    cl::Kernel kernel = opencl::get_kernel("compute_second_function");
    T delta = T(0.01);

    kernel.setArg(0, zeta.buffer());
    kernel.setArg(1, delta);
    kernel.setArg(2, 0); // t derivative
    kernel.setArg(3, dzeta.buffer());
    kernel.setArg(4, max_memory, NULL);
    kernel.setArg(5, nt);
    kernel.setArg(6, nx);
    kernel.setArg(7, ny);
    
    zeta.compute(kernel, blitz::shape(lt, lx, ly), blitz::shape(gt, gx, gy));
    dzeta.copy_to_host();

    T max_residual = 0;

    for (int i = 0; i < nt; i++)
    {
        using blitz::abs;
        using blitz::max;
        using blitz::min;
        using blitz::Range;
        Array2D<T> real_derivative =
            derivative<0, T>(zeta, arma::Vec3D<T>(0.01, 0.01, 0.01), i);
        T residual = min(abs(real_derivative - dzeta(i, Range::all(), Range::all())));
        max_residual = std::max(residual, max_residual);
    }
    EXPECT_NEAR(max_residual, T(0), T(1e-3));
}