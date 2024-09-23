include(AddTTGExecutable)

add_ttg_executable(tensortest madness/mra-device/tests/tensor_test.cc)
add_ttg_executable(mra-device-host
                   madness/mra-device/mra-ttg-device.cc
                   madness/mra-device/gl.cc
                   madness/mra-device/mra-ttg-device.cc
                   madness/mra-device/kernels.cc
                   madness/mratwoscale.cc
                   COMPILE_DEFINITIONS TTG_ENABLE_HOST=1
                   LINK_LIBRARIES MADworld
                   RUNTIMES "parsec")

add_ttg_executable(mra-device-host
                   madness/mra-device/mra-ttg-device.cc
                   madness/mra-device/gl.cc
                   madness/mra-device/mra-ttg-device.cc
                   madness/mra-device/kernels.cc
                   madness/mratwoscale.cc
                   COMPILE_DEFINITIONS TTG_ENABLE_HOST=1
                   LINK_LIBRARIES MADworld
                   RUNTIMES "mad")

if (TTG_HAVE_CUDA)
  add_ttg_executable(mradevice-cuda
                     madness/mra-device/mra-ttg-device.cc
                     madness/mra-device/gl.cc
                     madness/mra-device/mra-ttg-device.cc
                     madness/mra-device/kernels.cu
                     madness/mratwoscale.cc
                     COMPILE_DEFINITIONS TTG_ENABLE_CUDA=1
                     COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                     LINK_LIBRARIES MADworld
                     RUNTIMES "parsec")
endif (TTG_HAVE_CUDA)

if (TARGET MADworld)
  add_ttg_executable(madness1d madness/madness-1d/madness-1d.cc RUNTIMES "mad")
  if (TARGET blaspp) #(CBLAS_FOUND AND MKL_FOUND)
    add_ttg_executable(mra-ttg madness/madness-1d/mrattg.cc madness/madness-1d/mragl.cc madness/madness-1d/mratwoscale.cc madness/madness-1d/mradomain.h madness/madness-1d/mrafunctiondata.h madness/madness-1d/mrafunctionfunctor.h madness/madness-1d/mrafunctionnode.h madness/madness-1d/mragl.h madness/madness-1d/mrahash.h madness/madness-1d/mrakey.h madness/madness-1d/mramisc.h madness/madness-1d/mramxm.h madness/madness-1d/mrarange.h madness/madness-1d/mrasimpletensor.h madness/madness-1d/mratwoscale.h madness/madness-1d/mratypes.h LINK_LIBRARIES blaspp MADworld)

    add_ttg_executable(mra-ttg-streaming madness/madness-1d/mrattg_streaming.cc madness/madness-1d/mragl.cc madness/madness-1d/mratwoscale.cc madness/madness-1d/mradomain.h madness/madness-1d/mrafunctiondata.h madness/madness-1d/mrafunctionfunctor.h madness/madness-1d/mrafunctionnode.h madness/madness-1d/mragl.h madness/madness-1d/mrahash.h madness/madness-1d/mrakey.h madness/madness-1d/mramisc.h madness/madness-1d/mramxm.h madness/madness-1d/mrarange.h madness/madness-1d/mrasimpletensor.h madness/madness-1d/mratwoscale.h madness/madness-1d/mratypes.h LINK_LIBRARIES blaspp MADworld)
  endif ()
endif (TARGET MADworld)

