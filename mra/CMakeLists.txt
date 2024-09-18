include(AddTTGExecutable)

add_ttg_executable(tensortest madness/mra-device/tests/tensor_test.cc)
add_ttg_executable(mradevice-host
                   madness/mra-device/mrattg-device.cc
                   madness/mra-device/gl.cc
                   madness/mra-device/mrattg-device.cc
                   madness/mra-device/kernels.cc
                   mratwoscale.cc
                   COMPILE_DEFINITIONS TTG_ENABLE_HOST=1
                   LINK_LIBRARIES MADworld
                   RUNTIMES "parsec")
if (TTG_HAVE_CUDA)
  add_ttg_executable(mradevice-cuda
                     madness/mra-device/mrattg-device.cc
                     madness/mra-device/gl.cc
                     madness/mra-device/mrattg-device.cc
                     madness/mra-device/kernels.cu
                     mratwoscale.cc
                     COMPILE_DEFINITIONS TTG_ENABLE_CUDA=1
                     COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
                     LINK_LIBRARIES MADworld
                     RUNTIMES "parsec")
endif (TTG_HAVE_CUDA)

if (TARGET MADworld)
  #add_ttg_executable(madness-1d madness/madness-1d/madness-1d.cc RUNTIMES "mad")
  if (TARGET blaspp) #(CBLAS_FOUND AND MKL_FOUND)
    add_ttg_executable(mrattg madness/mrattg.cc mragl.cc mratwoscale.cc mradomain.h mrafunctiondata.h mrafunctionfunctor.h mrafunctionnode.h mragl.h mrahash.h mrakey.h mramisc.h mramxm.h mrarange.h mrasimpletensor.h mratwoscale.h mratypes.h LINK_LIBRARIES blaspp MADworld)

    add_ttg_executable(mrattg-streaming madness/mrattg_streaming.cc mragl.cc mratwoscale.cc mradomain.h mrafunctiondata.h mrafunctionfunctor.h mrafunctionnode.h mragl.h mrahash.h mrakey.h mramisc.h mramxm.h mrarange.h mrasimpletensor.h mratwoscale.h mratypes.h LINK_LIBRARIES blaspp MADworld)
  endif ()
endif (TARGET MADworld)

#add_ttg_executable(helloworld helloworld/helloworld.cpp)
