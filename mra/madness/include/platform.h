#ifndef MRA_DEVICE_PLATFORM_H
#define MRA_DEVICE_PLATFORM_H

#include <cstdlib>

/**
 * Utilities to achieve platform independence.
 * Provides wrappers for CUDA/HIP constructs to compile code
 * for both host and devices.
 */

namespace mra::detail {
  struct dim3
  {
      unsigned int x, y, z;
      constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz)
      { }
  };
} // namespace mra::detail

/* convenience macro to mark functions __device__ if compiling for CUDA */
#if defined(__CUDA_ARCH__)
#define SCOPE __device__ __host__
#define SYNCTHREADS() __syncthreads()
#define DEVSCOPE __device__
#define SHARED __shared__
#define LAUNCH_BOUNDS(__NT) __launch_bounds__(__NT)
#define HAVE_DEVICE_ARCH 1
#elif defined(__HIPCC__)
#define SCOPE __device__ __host__
#define SYNCTHREADS() __syncthreads()
#define DEVSCOPE __device__
#define SHARED __shared__
#define LAUNCH_BOUNDS(__NT) __launch_bounds__(__NT)
#define HAVE_DEVICE_ARCH 1
#else // __CUDA_ARCH__
#define SCOPE
#define SYNCTHREADS() do {} while(0)
#define DEVSCOPE inline
#define SHARED
#define LAUNCH_BOUNDS(__NT)
#endif // __CUDA_ARCH__

#ifdef HAVE_DEVICE_ARCH
using Dim3 = dim3;
#define GLOBALSCOPE __global__
#else // HAVE_DEVICE_ARCH
using Dim3 = mra::detail::dim3;
#define GLOBALSCOPE
#endif // HAVE_DEVICE_ARCH


#if defined(__CUDACC__)
#define checkSubmit() \
  if (cudaPeekAtLastError() != cudaSuccess)                         \
    std::cout << "kernel submission failed at " << __LINE__ << ": " \
    << cudaGetErrorString(cudaPeekAtLastError()) << std::endl;
#define CALL_KERNEL(name, block, thread, shared, stream, args) \
  name<<<block, thread, shared, stream>>> args
#elif defined(__HIPCC__)
#define checkSubmit() \
  if (hipPeekAtLastError() != hipSuccess)                           \
    std::cout << "kernel submission failed at " << __LINE__ << ": " \
    << hipGetErrorString(hipPeekAtLastError()) << std::endl;
#define CALL_KERNEL(name, block, thread, shared, stream, args) \
  name<<<block, thread, shared, stream>>> args
#else  // __CUDACC__
#define checkSubmit() do {} while(0)
#define CALL_KERNEL(name, blocks, thread, shared, stream, args) \
  do { \
    blockIdx = {0, 0, 0};                       \
    for (std::size_t i = 0; i < blocks; ++i) {  \
      blockIdx.x = i;                           \
      name args;                                \
    }                                           \
  } while (0)
#endif // __CUDACC__

#if defined(__CUDA_ARCH__)
#define THROW(s) do { std::printf(s); __trap(); } while(0)
#elif defined(__HIPCC__)
/* TODO: how to error out on HIP? */
#define THROW(s) do { printf(s); } while(0)
#else  // __CUDA_ARCH__
#define THROW(s) do { throw std::runtime_error(s); } while(0)
#endif // __CUDA_ARCH__


#if defined(MRA_ENABLE_HOST) || (defined(MRA_ENABLE_HIP) && !defined(HAVE_DEVICE_ARCH))
namespace mra {
  /* define our own thread layout (single thread) */
  static constexpr const detail::dim3 threadIdx = {0, 0, 0};
  static constexpr const detail::dim3 blockDim  = {1, 1, 1};
  static constexpr const detail::dim3 gridDim   = {1, 1, 1};
  inline thread_local    detail::dim3 blockIdx  = {0, 0, 0};
} // namespace mra
#endif // MRA_ENABLE_HOST

/**
 * Function returning the thread ID in a flat ID space.
 */
namespace mra {
  DEVSCOPE int thread_id() {
    return blockDim.x * ((blockDim.y * threadIdx.z) + threadIdx.y) + threadIdx.x;
  }

  DEVSCOPE int block_size() {
    return blockDim.x * blockDim.y * blockDim.z;
  }
} // namespace mra

#endif // MRA_DEVICE_PLATFORM_H
