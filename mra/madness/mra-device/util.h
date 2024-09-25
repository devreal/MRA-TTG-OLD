#ifndef MRA_DEVICE_UTIL_H
#define MRA_DEVICE_UTIL_H

/* convenience macro to mark functions __device__ if compiling for CUDA */
#if defined(__CUDA_ARCH__)
#define SCOPE __device__ __host__
#define VARSCOPE __device__
#define SYNCTHREADS() __syncthreads()
#define DEVSCOPE __device__
#define SHARED __shared
#else // __CUDA_ARCH__
#define SCOPE
#define VARSCOPE
#define SYNCTHREADS()
#define DEVSCOPE
#define SHARED
#endif // __CUDA_ARCH__

#ifdef __CUDACC__
#define checkSubmit() \
  if (cudaPeekAtLastError() != cudaSuccess)                         \
    std::cout << "kernel submission failed at " << __LINE__ << ": " \
    << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
#define CALL_KERNEL(name, ...) name<<<__VA_ARGS__>>>
#else  // __CUDACC__
#define checkSubmit()
#define CALL_KERNEL(name, ...) name
#endif // __CUDACC__

#endif // MRA_DEVICE_UTIL_H