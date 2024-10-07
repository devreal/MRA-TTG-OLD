include(AddTTGExecutable)

add_ttg_executable(mra-tensortest madness/mra-device/tests/tensor_test.cc)
add_ttg_executable(mra-device-host
                   madness/mra-device/mra-ttg-device.cc
                   madness/mra-device/gl.cc
                   madness/mra-device/kernels.cc
                   madness/mratwoscale.cc
                   COMPILE_DEFINITIONS TTG_ENABLE_HOST=1
                   LINK_LIBRARIES MADworld
                   RUNTIMES "parsec")

add_ttg_executable(mra-device-uot-host
                   madness/mra-device/mra-ttg-uot-device.cc
                   madness/mra-device/gl.cc
                   madness/mra-device/kernels.cc
                   madness/mratwoscale.cc
                   COMPILE_DEFINITIONS TTG_ENABLE_HOST=1
                   LINK_LIBRARIES MADworld
                   RUNTIMES "parsec")

if (TTG_HAVE_CUDA)
  add_ttg_executable(mra-device-cuda
                     madness/mra-device/mra-ttg-device.cc
                     madness/mra-device/gl.cc
                     madness/mra-device/kernels.cu
                     madness/mratwoscale.cc
                     COMPILE_DEFINITIONS TTG_ENABLE_CUDA=1
                     COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                     LINK_LIBRARIES MADworld
                     RUNTIMES "parsec")

  add_ttg_executable(mra-device-uot-cuda
                     madness/mra-device/mra-ttg-uot-device.cc
                     madness/mra-device/gl.cc
                     madness/mra-device/kernels.cu
                     madness/mratwoscale.cc
                     COMPILE_DEFINITIONS TTG_ENABLE_CUDA=1
                     COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                     LINK_LIBRARIES MADworld
                     RUNTIMES "parsec")
endif (TTG_HAVE_CUDA)


