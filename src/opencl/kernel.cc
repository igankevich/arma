#include "kernel.hh"
#include <iostream>

void
arma::opencl::Kernel::run() {
	cl_event evt;
	cl_int err = CL_SUCCESS;
	const size_t* local_work_size =
		_localsize.empty()
		? nullptr
		: _localsize.data();
	err |= clEnqueueNDRangeKernel(
		command_queue(),
		_kernel,
		_globalsize.size(), 0,
		_globalsize.data(),
		local_work_size,
		0, 0, &evt
	);
	err |= clWaitForEvents(1, &evt);
	err |= clReleaseEvent(evt);
	check_err(err, _name.data());
}

std::string
arma::opencl::Kernel::get_function_name(cl_kernel kernel) {
	char funcName[4096] = {0};
	clGetKernelInfo(
		kernel,
		CL_KERNEL_FUNCTION_NAME,
		sizeof(funcName),
		funcName,
		0
	);
	return std::string(funcName);
}

void
arma::opencl::Kernel::check(cl_int err, cl_int arg) {
	if (err != CL_SUCCESS) {
		std::cerr << "OpenCL error=" << err
			<< ",kernel=" << this->_name
			<< ",argument=" << arg
			<< std::endl;
		check_err(err, "kernel error");
		std::exit(err);
	}
}
