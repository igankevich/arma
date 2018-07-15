#include "opencl.hh"
#include <GL/gl.h>
#include <clFFT.h>
#include "high_amplitude_realtime_solver.hh"
#include "opencl/vec.hh"
#include "profile.hh"
#include "derivative.hh"
#include <gtest/gtest.h>
#include <cmath>
#include "array.hh"
#include "discrete_function.hh"
#include "cl.hh"
#include "device_type.hh"

using namespace arma;

typedef double T;

arma::Array3D<T> fill_sin(int nx, int ny, int nz)
{
    arma::Array3D<T> zeta(nx, ny, nz);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            for (int k = 0; k < nz; k++)
                zeta(i, j, k) = sin(T(i + j +k) / T(100));
    return zeta;
}


TEST(Derivatives, Accuracy) {
    ::arma::opencl::init();
    const int nx = 30;
    const int ny = 20;
    const int nz = 25;
    arma::Array3D<T> zeta = fill_sin(nx, ny, nz);

    cl::Buffer bzeta(
        opencl::context(),
        const_cast<T *>(zeta.data()),
        const_cast<T *>(zeta.data() + zeta.numElements()),
        CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR);
    T *data = new T[nx * ny * nz];
    cl::Buffer _phi(opencl::context(),
                    data,
                    data + nx * ny * nz,
                    CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR);
    size_t max_memory = opencl::devices()[0].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    cl::Kernel kernel = opencl::get_kernel("compute_second_function");
    T delta[] = {T(0.01), T(0.01), T(0.01)};
    cl::Buffer bdelta(
        opencl::context(),
        delta,
        delta + 3,
        CL_MEM_READ_WRITE, CL_MEM_USE_HOST_PTR);

    kernel.setArg(0, bzeta);
    kernel.setArg(1, bdelta);
    kernel.setArg(2, 0); // t derivative
    kernel.setArg(3, _phi);
    kernel.setArg(4, max_memory, NULL);

    opencl::command_queue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        cl::NDRange(nx, ny, nz));

    opencl::command_queue().enqueueReadBuffer(_phi, CL_TRUE, 0, nx * ny * nz, data);
    T max = 0;

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nz; k++)
            {
                Array2D<T> real_derivative = 
                    derivative<0, T>(zeta, arma::Vec3D<T>(0.01, 0.01, 0.01), i);
                int id = i * ny * nz + j * nz + k;
                if (std::isnan(data[id])) std::cout << id << std::endl;
                max = std::max(std::abs(data[id] - real_derivative(j, k)), max);
            }
        }
    }
    EXPECT_NEAR(max, T(0), T(1e-3));
}


int main(int argc, char *argv[])
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
