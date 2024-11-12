include(AddTTGExecutable)

include_directories(madness/include)

add_ttg_executable(mra-tensortest madness/tests/tensor_test.cc)
add_ttg_executable(mra-device-host
                   madness/src/mra-ttg-device.cc
                   madness/src/gl.cc
                   madness/src/kernels.cc
                   madness/src/mratwoscale.cc
                   COMPILE_DEFINITIONS TTG_ENABLE_HOST=1
                   LINK_LIBRARIES MADworld
                   RUNTIMES "parsec")

add_ttg_executable(mra-device-uot-host
                   madness/src/mra-ttg-uot-device.cc
                   madness/src/gl.cc
                   madness/src/kernels.cc
                   madness/src/mratwoscale.cc
                   COMPILE_DEFINITIONS TTG_ENABLE_HOST=1
                   LINK_LIBRARIES MADworld
                   RUNTIMES "parsec")

if (HAVE_CUDA)
  add_ttg_executable(mra-device-cuda
                     madness/src/mra-ttg-device.cc
                     madness/src/gl.cc
                     madness/src/kernels.cu
                     madness/src/mratwoscale.cc
                     COMPILE_DEFINITIONS TTG_ENABLE_CUDA=1
                     COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                     LINK_LIBRARIES MADworld
                     RUNTIMES "parsec")

  add_ttg_executable(mra-device-uot-cuda
                     madness/src/mra-ttg-uot-device.cc
                     madness/src/gl.cc
                     madness/src/kernels.cu
                     madness/src/mratwoscale.cc
                     COMPILE_DEFINITIONS TTG_ENABLE_CUDA=1
                     COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                     LINK_LIBRARIES MADworld
                     RUNTIMES "parsec")
endif (HAVE_CUDA)


